"""Validate S2-A.1 activation summaries or a complete evaluation artifact set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_VARIANTS = (
    "S2A1_C0_BASELINE",
    "S2A1_C1_FORCED_REFRESH",
    "S2A1_C2_LOCAL_CONNECTOR",
)
RECOVERY_SCHEMA = "bser.phase1c.prrac.search_collision_recovery.v2"
ACTIVATION_SCHEMA = "bser.phase1c.prrac.search_collision_recovery.activation.v2"
REPORT_SCHEMA = "bser.phase1c.prrac.evaluation_report.v2"
PROGRESS_SCHEMA = "bser.phase1c.prrac.evaluation_progress.v2"
ACTIVATION_ARTIFACT_REVISION = "s2a1.activation_artifact.v1"
NATIVE_RUNTIME_REVISION = "dynamic_public_intercept_v3_atomic_continuity"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _integer(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name, "")
    return 0 if value in {"", "None", None} else int(float(value))


def _boolean(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _result(checks: Mapping[str, bool], counts: Mapping[str, int], failures: Iterable[str]) -> dict[str, object]:
    messages = list(failures)
    return {
        "status": "PASS" if all(checks.values()) and not messages else "FAIL",
        "checks": dict(checks),
        "counts": dict(counts),
        "failures": messages,
    }


def validate_activation(summary_csv: Path) -> dict[str, object]:
    """Retain the historical summary-only weak check."""

    rows = _read_csv(Path(summary_csv))
    by_variant = {str(row.get("search_recovery_variant")): row for row in rows}
    checks: dict[str, bool] = {"variants_complete": set(REQUIRED_VARIANTS) <= set(by_variant)}
    failures: list[str] = []
    if not checks["variants_complete"]:
        failures.append("activation summary does not contain all three S2-A.1 variants")
        return _result(checks, {"summary_rows": len(rows)}, failures)
    c0, c1, c2 = (by_variant[name] for name in REQUIRED_VARIANTS)
    no_op_fields = (
        "search_recovery_entry_count", "forced_public_refresh_count",
        "recovery_plan_active_step_count", "recovery_guidance_changed_step_count",
        "recovery_effective_intervention_count", "local_connector_attempt_count",
    )
    checks["c0_strict_no_op"] = all(_integer(c0, field) == 0 for field in no_op_fields)
    checks["recovery_entries_exist"] = sum(_integer(row, "search_recovery_entry_count") for row in (c1, c2)) > 0
    checks["c2_local_connector_attempt"] = _integer(c2, "local_connector_attempt_count") > 0
    checks["c2_local_connector_plan"] = _integer(c2, "local_connector_plan_count") > 0
    checks["guidance_changed"] = any(_integer(row, "recovery_guidance_changed_step_count") > 0 for row in (c1, c2))
    checks["plan_active"] = any(_integer(row, "recovery_plan_active_step_count") > 0 for row in (c1, c2))
    checks["effective_episode"] = any(_integer(row, "recovery_effective_intervention_episode_count") > 0 for row in (c1, c2))
    checks["single_manifest"] = len({row.get("manifest_sha256") for row in by_variant.values()}) == 1
    checks["v2_schema"] = all(row.get("search_collision_recovery_schema") == RECOVERY_SCHEMA for row in by_variant.values())
    failures.extend(name for name, passed in checks.items() if not passed)
    return _result(checks, {"summary_rows": len(rows)}, failures)


def validate_output_dir(output_dir: Path) -> dict[str, object]:
    output = Path(output_dir)
    names = {
        "config": "resolved_evaluation_config.json",
        "manifest": "evaluation_manifest.json",
        "progress": "evaluation_progress.json",
        "episodes": "search_collision_recovery_episode.csv",
        "summaries": "search_collision_recovery_summary.csv",
        "activation": "search_collision_recovery_activation_steps.csv",
    }
    checks: dict[str, bool] = {}
    counts: dict[str, int] = {}
    failures: list[str] = []
    paths = {key: output / value for key, value in names.items()}
    checks["required_artifacts_present"] = all(path.is_file() for path in paths.values())
    if not checks["required_artifacts_present"]:
        failures.extend(f"missing artifact: {path.name}" for path in paths.values() if not path.is_file())
        return _result(checks, counts, failures)
    try:
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
        episodes = _read_csv(paths["episodes"])
        summaries = _read_csv(paths["summaries"])
        activation = _read_csv(paths["activation"])
    except (OSError, ValueError, json.JSONDecodeError) as error:
        checks["artifacts_parse"] = False
        failures.append(f"artifact parse failure: {error}")
        return _result(checks, counts, failures)
    checks["artifacts_parse"] = True
    counts.update({"episode_rows": len(episodes), "summary_rows": len(summaries), "activation_rows": len(activation)})

    scenario_rows = list(manifest.get("scenarios", ()))
    scenario_pairs = [(str(row.get("scenario_id", "")), int(row.get("scenario_seed", -1))) for row in scenario_rows]
    scenario_map = dict(scenario_pairs)
    scenario_count = len(scenario_rows)
    counts["scenario_count"] = scenario_count
    checks["scenario_manifest_unique"] = bool(scenario_rows) and len(scenario_map) == scenario_count
    count_values = (
        manifest.get("evaluation_episodes"), manifest.get("selected_scenario_count"),
        manifest.get("resolved_evaluation_episodes"), config.get("evaluation_episodes"),
        config.get("selected_scenario_count"), config.get("resolved_evaluation_episodes"),
        progress.get("selected_scenario_count"), progress.get("resolved_evaluation_episodes"),
    )
    checks["resolved_counts_match"] = all(value is not None and int(value) == scenario_count for value in count_values)
    checks["generated_count_contract"] = (
        int(config.get("requested_evaluation_episodes", -1))
        == int(manifest.get("requested_evaluation_episodes", -2))
        and int(config.get("generated_scenario_count", -1))
        == int(manifest.get("generated_scenario_count", -2))
        >= scenario_count
    )
    targeted = str(manifest.get("scenario_selection_mode")) == "baseline_collision_targeted_smoke"
    checks["targeted_flags_consistent"] = (
        all(_boolean(item.get("diagnostic_only")) for item in (config, manifest, progress))
        and str(config.get("scenario_selection_mode")) == "baseline_collision_targeted_smoke"
        and str(progress.get("scenario_selection_mode")) == "baseline_collision_targeted_smoke"
    ) if targeted else (
        not any(_boolean(item.get("diagnostic_only")) for item in (config, manifest, progress))
        and str(manifest.get("scenario_selection_mode")) == "generated_manifest_order"
    )

    expected = {
        "manifest_sha256": str(config.get("manifest_sha256", "")),
        "search_collision_recovery_schema": RECOVERY_SCHEMA,
        "search_collision_recovery_config_hash": str(config.get("search_collision_recovery_config_hash", "")),
        "activation_diagnostics_schema": ACTIVATION_SCHEMA,
        "activation_artifact_revision": ACTIVATION_ARTIFACT_REVISION,
        "s2a1_activation_artifact_revision": ACTIVATION_ARTIFACT_REVISION,
        "report_schema": REPORT_SCHEMA,
    }
    checks["top_level_schema_provenance"] = (
        config.get("schema") == REPORT_SCHEMA
        and progress.get("schema") == PROGRESS_SCHEMA
        and manifest.get("manifest_sha256") == expected["manifest_sha256"]
        and all(str(progress.get(key, "")) == value for key, value in expected.items())
    )
    provenance_fields = tuple(expected)
    evaluation_revisions = set(map(str, config.get("resolved_evaluation_runtime_revisions", ())))
    integration_modes = set(map(str, config.get("resolved_runtime_integration_modes", ())))
    checks["native_b1_protocol_identity"] = (
        len(config.get("resolved_checkpoint_paths", ())) == 1
        and set(map(str, config.get("resolved_evaluation_modes", ()))) == {"full_prrac"}
        and set(map(str, config.get("resolved_execution_variants", ()))) == {"B1_ATOMIC_LAST_VALID"}
        and str(config.get("checkpoint_runtime_revision")) == NATIVE_RUNTIME_REVISION
        and evaluation_revisions == {NATIVE_RUNTIME_REVISION}
        and integration_modes == {"native"}
    )
    checks["episode_provenance"] = all(
        all(str(row.get(field, "")) == expected[field] for field in provenance_fields)
        and str(row.get("checkpoint_runtime_revision", "")) == str(config.get("checkpoint_runtime_revision", ""))
        and str(row.get("evaluation_runtime_revision", "")) in evaluation_revisions
        and str(row.get("runtime_integration_mode", "")) in integration_modes
        for row in episodes
    )
    checks["checkpoint_provenance_unique"] = (
        len({str(row.get("checkpoint")) for row in episodes}) == 1
        and len({str(row.get("checkpoint_episode")) for row in episodes}) == 1
        and len({str(row.get("checkpoint_config_hash")) for row in episodes}) == 1
        and all(str(row.get("evaluation_mode")) == "full_prrac" for row in episodes)
        and all(str(row.get("execution_variant")) == "B1_ATOMIC_LAST_VALID" for row in episodes)
    )

    def group_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (str(row.get("checkpoint")), str(row.get("evaluation_mode")), str(row.get("execution_variant")))

    episode_keys = {
        (*group_key(row), str(row.get("search_recovery_variant")), str(row.get("scenario_id")))
        for row in episodes
    }
    checks["episode_rows_unique"] = len(episode_keys) == len(episodes)
    episode_groups: dict[tuple[str, str, str], dict[str, list[dict[str, str]]]] = {}
    for row in episodes:
        episode_groups.setdefault(group_key(row), {}).setdefault(str(row.get("search_recovery_variant")), []).append(row)
    expected_groups = {
        (str(checkpoint), str(mode), str(variant))
        for checkpoint in config.get("resolved_checkpoint_paths", ())
        for mode in config.get("resolved_evaluation_modes", ())
        for variant in config.get("resolved_execution_variants", ())
    }
    checks["exact_evaluation_groups"] = set(episode_groups) == expected_groups
    checks["variants_exact"] = all(set(group) == set(REQUIRED_VARIANTS) for group in episode_groups.values())
    checks["scenario_ids_and_seeds_paired"] = all(
        {str(row.get("scenario_id")): int(row.get("scenario_seed", -1)) for row in values} == scenario_map
        and len(values) == scenario_count
        for group in episode_groups.values() for values in group.values()
    )

    summary_keys = [(*group_key(row), str(row.get("search_recovery_variant"))) for row in summaries]
    expected_summary_keys = {(*key, variant) for key in expected_groups for variant in REQUIRED_VARIANTS}
    checks["summaries_exact"] = (
        len(set(summary_keys)) == len(summary_keys)
        and set(summary_keys) == expected_summary_keys
        and all(_integer(row, "evaluation_episodes") == scenario_count for row in summaries)
        and all(all(str(row.get(field, "")) == expected[field] for field in provenance_fields) for row in summaries)
    )
    summary_by_variant = {variant: [row for row in summaries if row.get("search_recovery_variant") == variant] for variant in REQUIRED_VARIANTS}
    no_op = (
        "search_recovery_entry_count", "forced_public_refresh_count",
        "recovery_plan_active_step_count", "recovery_guidance_changed_step_count",
        "recovery_effective_intervention_count", "local_connector_attempt_count",
    )
    checks["c0_strict_no_op"] = all(_integer(row, field) == 0 for row in summary_by_variant[REQUIRED_VARIANTS[0]] for field in no_op)
    checks["c1_c2_entries"] = all(sum(_integer(row, "search_recovery_entry_count") for row in summary_by_variant[variant]) > 0 for variant in REQUIRED_VARIANTS[1:])
    checks["c2_local_attempt_and_plan"] = (
        sum(_integer(row, "local_connector_attempt_count") for row in summary_by_variant[REQUIRED_VARIANTS[2]]) > 0
        and sum(_integer(row, "local_connector_plan_count") for row in summary_by_variant[REQUIRED_VARIANTS[2]]) > 0
    )
    active_rows = summary_by_variant[REQUIRED_VARIANTS[1]] + summary_by_variant[REQUIRED_VARIANTS[2]]
    checks["active_intervention"] = (
        sum(_integer(row, "recovery_plan_active_step_count") for row in active_rows) > 0
        and sum(_integer(row, "recovery_guidance_changed_step_count") for row in active_rows) > 0
        and sum(_integer(row, "recovery_effective_intervention_episode_count") for row in active_rows) > 0
    )

    activation_keys = {
        (str(row.get("checkpoint")), str(row.get("search_recovery_variant")), str(row.get("scenario_id")),
         _integer(row, "step"), _integer(row, "agent_id"), _integer(row, "attempt_id"))
        for row in activation
    }
    checks["activation_rows_unique"] = len(activation_keys) == len(activation)
    checks["c0_has_zero_activation_rows"] = not any(row.get("search_recovery_variant") == REQUIRED_VARIANTS[0] for row in activation)
    checks["activation_scope"] = all(
        row.get("search_recovery_variant") in REQUIRED_VARIANTS[1:]
        and (str(row.get("recovery_mode")) != "NORMAL_SEARCH" or _boolean(row.get("recovery_plan_installed")) or _boolean(row.get("guidance_changed")))
        for row in activation
    )
    checks["activation_provenance"] = all(
        all(str(row.get(field, "")) == expected[field] for field in provenance_fields)
        and str(row.get("scenario_id")) in scenario_map
        and int(row.get("scenario_seed", -1)) == scenario_map[str(row.get("scenario_id"))]
        and str(row.get("checkpoint")) in {str(item.get("checkpoint")) for item in episodes}
        and str(row.get("checkpoint_episode")) in {str(item.get("checkpoint_episode")) for item in episodes}
        and str(row.get("checkpoint_config_hash")) in {str(item.get("checkpoint_config_hash")) for item in episodes}
        and str(row.get("checkpoint_runtime_revision")) == NATIVE_RUNTIME_REVISION
        and str(row.get("evaluation_runtime_revision")) == NATIVE_RUNTIME_REVISION
        and str(row.get("runtime_integration_mode")) == "native"
        and str(row.get("execution_variant")) == "B1_ATOMIC_LAST_VALID"
        and str(row.get("evaluation_mode")) == "full_prrac"
        for row in activation
    )
    checks["activation_change_exists"] = any(
        _boolean(row.get("guidance_changed"))
        and (_boolean(row.get("path_changed")) or float(row.get("tracking_waypoint_delta_norm") or 0.0) > 0.0)
        for row in activation
    )

    completed = list(progress.get("completed", ()))
    completed_keys = [
        (str(row.get("checkpoint")), str(row.get("evaluation_mode")), str(row.get("execution_variant")), str(row.get("search_recovery_variant")))
        for row in completed
    ]
    checks["progress_combinations_exact"] = (
        len(set(completed_keys)) == len(completed_keys)
        and set(completed_keys) == expected_summary_keys
        and all(all(str(row.get(field, "")) == expected[field] for field in provenance_fields) for row in completed)
    )
    failures.extend(name for name, passed in checks.items() if not passed)
    return _result(checks, counts, failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--output-dir", type=Path)
    source.add_argument("--summary-csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = validate_output_dir(args.output_dir) if args.output_dir else validate_activation(args.summary_csv)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
