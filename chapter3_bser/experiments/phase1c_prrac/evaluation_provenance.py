"""Strict row-derived provenance and cross-artifact consistency checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


EVALUATION_REPORT_SCHEMA = "bser.phase1c.prrac.evaluation_report.v2"
EVALUATION_SUMMARY_SCHEMA = "bser.phase1c.prrac.evaluation_summary.v2"
EXECUTION_SUMMARY_SCHEMA = "bser.phase1c.prrac.execution_summary.v2"
SEARCH_SUMMARY_SCHEMA = "bser.phase1c.prrac.search_summary.v2"
PROGRESS_SCHEMA = "bser.phase1c.prrac.evaluation_progress.v2"

PROVENANCE_FIELDS = (
    "checkpoint",
    "checkpoint_episode",
    "checkpoint_config_hash",
    "checkpoint_runtime_revision",
    "evaluation_runtime_revision",
    "runtime_integration_mode",
    "execution_variant",
    "evaluation_mode",
    "manifest_sha256",
    "search_continuity_diagnostics_hash",
    "search_recovery_variant",
    "search_collision_recovery_schema",
    "search_collision_recovery_config_hash",
    "activation_diagnostics_schema",
    "report_schema",
)


def _normal(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if value is None:
        return ""
    return value


def derive_unique_provenance(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return scalar values only when unique and always expose sorted plural values."""

    values = list(rows)
    result: dict[str, Any] = {}
    for field in PROVENANCE_FIELDS:
        unique = sorted({_normal(row.get(field)) for row in values if row.get(field) is not None}, key=str)
        result[field] = unique[0] if len(unique) == 1 else None
        result[f"{field}_values"] = unique
    return result


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if _normal(actual) != _normal(expected):
        raise ValueError(f"evaluation provenance mismatch for {label}: {actual!r} != {expected!r}")


def validate_evaluation_provenance(
    *,
    rows: list[Mapping[str, Any]],
    resolved_config: Mapping[str, Any],
    progress: Mapping[str, Any],
    checkpoint_metadata: Iterable[Mapping[str, Any]],
    summary_groups: Iterable[Mapping[str, Any]] = (),
    expected_scenarios: Iterable[Mapping[str, Any]],
) -> None:
    """Fail closed before final report writers see inconsistent artifacts."""

    if str(resolved_config.get("schema")) != EVALUATION_REPORT_SCHEMA:
        raise ValueError("evaluation report schema mismatch")
    if str(progress.get("schema")) != PROGRESS_SCHEMA:
        raise ValueError("evaluation progress schema mismatch")
    _require_equal("progress manifest_sha256", progress.get("manifest_sha256"), resolved_config.get("manifest_sha256"))
    for field in (
        "search_collision_recovery_schema", "activation_diagnostics_schema",
        "activation_artifact_revision", "report_schema",
        "s2a1_activation_artifact_revision",
    ):
        _require_equal(f"progress {field}", progress.get(field), resolved_config.get(field))
    expected_values = list(expected_scenarios)
    expected_ids = {str(row["scenario_id"]) for row in expected_values}
    expected_seeds = {int(row["scenario_seed"]) for row in expected_values}
    seed_complete = all(row.get("scenario_seed") is not None for row in expected_values)
    expected_pairs = {
        (str(row["scenario_id"]), int(row["scenario_seed"]))
        for row in expected_values if row.get("scenario_seed") is not None
    }
    _require_equal(
        "resolved evaluation episode count",
        resolved_config.get("resolved_evaluation_episodes"),
        len(expected_values),
    )
    _require_equal(
        "progress evaluation episode count",
        progress.get("resolved_evaluation_episodes"),
        len(expected_values),
    )
    if rows:
        row_ids = {str(row["scenario_id"]) for row in rows}
        row_seeds = {int(row["scenario_seed"]) for row in rows}
        if row_ids != expected_ids:
            raise ValueError("evaluation provenance scenario_id set mismatch")
        if row_seeds != expected_seeds:
            raise ValueError("evaluation provenance scenario_seed set mismatch")
    resolved_checkpoints = {str(Path(value).resolve()) for value in resolved_config.get("resolved_checkpoint_paths", ())}
    metadata_by_path = {str(Path(item["checkpoint"]).resolve()): item for item in checkpoint_metadata}
    if resolved_checkpoints != set(metadata_by_path):
        raise ValueError("evaluation provenance checkpoint metadata path set mismatch")
    expected_fields = {
        "manifest_sha256": resolved_config.get("manifest_sha256"),
        "search_continuity_diagnostics_hash": resolved_config.get("search_continuity_diagnostics_hash"),
        "search_collision_recovery_config_hash": resolved_config.get("search_collision_recovery_config_hash"),
        "search_collision_recovery_schema": resolved_config.get("search_collision_recovery_schema"),
        "activation_diagnostics_schema": resolved_config.get("activation_diagnostics_schema"),
        "report_schema": resolved_config.get("schema"),
        "checkpoint_runtime_revision": resolved_config.get("checkpoint_runtime_revision"),
    }
    allowed = {
        "evaluation_mode": set(resolved_config.get("resolved_evaluation_modes", ())),
        "execution_variant": set(resolved_config.get("resolved_execution_variants", ())),
        "search_recovery_variant": set(resolved_config.get("resolved_search_recovery_variants", ())),
        "evaluation_runtime_revision": set(
            resolved_config.get(
                "resolved_evaluation_runtime_revisions",
                (resolved_config.get("evaluation_runtime_revision"),),
            )
        ),
        "runtime_integration_mode": set(
            resolved_config.get(
                "resolved_runtime_integration_modes",
                (resolved_config.get("runtime_integration_mode"),),
            )
        ),
    }
    combo_fields = (
        "checkpoint", "checkpoint_config_hash", "checkpoint_episode",
        "checkpoint_runtime_revision", "evaluation_runtime_revision",
        "runtime_integration_mode", "execution_variant", "evaluation_mode",
        "search_recovery_variant", "manifest_sha256",
        "execution_overlay_config_hash", "search_continuity_diagnostics_hash",
        "search_collision_recovery_schema", "search_collision_recovery_config_hash",
        "activation_diagnostics_schema", "activation_artifact_revision", "report_schema",
        "s2a1_activation_artifact_revision",
    )
    row_combo_keys = {
        tuple(_normal(row.get(field)) for field in combo_fields) for row in rows
    }
    progress_values = list(progress.get("completed", ()))
    progress_combo_keys = {
        tuple(_normal(combo.get(field)) for field in combo_fields)
        for combo in progress_values
    }
    if len(progress_combo_keys) != len(progress_values):
        raise ValueError("evaluation progress contains duplicate completed combinations")
    if progress_combo_keys != row_combo_keys:
        raise ValueError("evaluation progress completed combinations do not exactly match episode rows")
    expected_combo_count = (
        len(resolved_checkpoints)
        * len(allowed["evaluation_mode"])
        * len(allowed["execution_variant"])
        * len(allowed["search_recovery_variant"])
    )
    if len(row_combo_keys) != expected_combo_count:
        raise ValueError("evaluation completed combination count mismatch")
    episode_identity = {
        (*tuple(_normal(row.get(field)) for field in combo_fields), str(row.get("scenario_id")))
        for row in rows
    }
    if len(episode_identity) != len(rows):
        raise ValueError("evaluation episode rows contain duplicate combination/scenario identities")
    for index, row in enumerate(rows):
        checkpoint = str(Path(row["checkpoint"]).resolve())
        if checkpoint not in metadata_by_path:
            raise ValueError(f"episode row {index} has unregistered checkpoint")
        metadata_item = metadata_by_path[checkpoint]
        metadata = dict(metadata_item.get("metadata", {}))
        _require_equal(f"row {index} checkpoint_episode", row.get("checkpoint_episode"), metadata_item.get("completed_episode"))
        _require_equal(f"row {index} checkpoint_config_hash", row.get("checkpoint_config_hash"), metadata.get("config_hash", ""))
        _require_equal(
            f"row {index} checkpoint metadata runtime",
            row.get("checkpoint_runtime_revision"),
            metadata.get("execution_runtime_revision", ""),
        )
        for field, expected in expected_fields.items():
            _require_equal(f"row {index} {field}", row.get(field), expected)
        for field, choices in allowed.items():
            if row.get(field) not in choices:
                raise ValueError(f"episode row {index} has unregistered {field}: {row.get(field)!r}")
    for summary in summary_groups:
        matching = [
            row for row in rows
            if all(row.get(field) == summary.get(field) for field in ("checkpoint", "evaluation_mode", "execution_variant", "search_recovery_variant", "manifest_sha256"))
        ]
        if not matching:
            raise ValueError("summary provenance does not identify episode rows")
        matching_pairs = {
            (str(row["scenario_id"]), int(row["scenario_seed"])) for row in matching
        }
        if matching_pairs != expected_pairs or len(matching) != len(expected_pairs):
            raise ValueError("summary provenance scenario count/id/seed mismatch")
        derived = derive_unique_provenance(matching)
        for field in PROVENANCE_FIELDS:
            if summary.get(field) is not None:
                _require_equal(f"summary {field}", summary.get(field), derived.get(field))
    for combo in progress_values:
        matching = [
            row
            for row in rows
            if all(
                _normal(row.get(field)) == _normal(combo.get(field))
                for field in (
                    *combo_fields,
                )
            )
        ]
        ids = {str(row["scenario_id"]) for row in matching}
        pairs = {(str(row["scenario_id"]), int(row["scenario_seed"])) for row in matching if row.get("scenario_seed") is not None}
        if ids != expected_ids or len(matching) != len(expected_ids) or (seed_complete and pairs != expected_pairs):
            raise ValueError("evaluation progress combo scenario count/id/seed mismatch")


def validate_summary_provenance(
    rows: list[Mapping[str, Any]],
    summaries: Iterable[Mapping[str, Any]],
    expected_scenarios: Iterable[Mapping[str, Any]],
) -> None:
    expected_values = list(expected_scenarios)
    expected_ids = {str(row["scenario_id"]) for row in expected_values}
    seed_complete = all(row.get("scenario_seed") is not None for row in expected_values)
    expected_pairs = {
        (str(row["scenario_id"]), int(row["scenario_seed"]))
        for row in expected_values if row.get("scenario_seed") is not None
    }
    identity = (
        "checkpoint", "checkpoint_config_hash", "checkpoint_episode",
        "checkpoint_runtime_revision", "evaluation_runtime_revision",
        "runtime_integration_mode", "execution_variant", "evaluation_mode",
        "search_recovery_variant", "manifest_sha256",
        "search_continuity_diagnostics_hash", "search_collision_recovery_schema",
        "search_collision_recovery_config_hash", "activation_diagnostics_schema",
        "report_schema",
    )
    for summary in summaries:
        matching = [
            row for row in rows
            if all(summary.get(field) is None or row.get(field) == summary.get(field) for field in identity)
        ]
        ids = {str(row["scenario_id"]) for row in matching}
        pairs = {(str(row["scenario_id"]), int(row["scenario_seed"])) for row in matching if row.get("scenario_seed") is not None}
        if ids != expected_ids or len(matching) != len(expected_ids) or (seed_complete and pairs != expected_pairs):
            raise ValueError("report summary scenario/provenance group mismatch")
        derived = derive_unique_provenance(matching)
        for field in PROVENANCE_FIELDS:
            if summary.get(field) is not None:
                _require_equal(f"report summary {field}", summary.get(field), derived.get(field))


def validate_resume_config(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field in (
        "schema", "resolved_config_hash", "manifest_sha256",
        "search_collision_recovery_schema", "search_collision_recovery_config_hash",
        "activation_diagnostics_schema", "activation_artifact_revision",
        "s2a1_activation_artifact_revision",
        "search_continuity_diagnostics_hash", "report_schema",
    ):
        _require_equal(f"resume {field}", saved.get(field), expected.get(field))
    _require_equal("resume resolved scenario ids", saved.get("resolved_scenario_ids"), expected.get("resolved_scenario_ids"))


__all__ = (
    "EVALUATION_REPORT_SCHEMA", "EVALUATION_SUMMARY_SCHEMA", "EXECUTION_SUMMARY_SCHEMA",
    "PROGRESS_SCHEMA", "PROVENANCE_FIELDS", "SEARCH_SUMMARY_SCHEMA",
    "derive_unique_provenance", "validate_evaluation_provenance", "validate_resume_config",
    "validate_summary_provenance",
)
