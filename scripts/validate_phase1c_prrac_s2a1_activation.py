"""Validate that S2-A.1 recovery entered the closed-loop guidance path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _integer(row: dict[str, str], name: str) -> int:
    value = row.get(name, "")
    return 0 if value in {"", "None", None} else int(float(value))


def validate_activation(summary_csv: Path) -> dict[str, object]:
    with Path(summary_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_variant = {str(row.get("search_recovery_variant")): row for row in rows}
    required = {"S2A1_C0_BASELINE", "S2A1_C1_FORCED_REFRESH", "S2A1_C2_LOCAL_CONNECTOR"}
    checks: dict[str, bool] = {"variants_complete": required <= set(by_variant)}
    if not checks["variants_complete"]:
        return {"status": "FAIL", "checks": checks}
    c0, c1, c2 = (by_variant[name] for name in ("S2A1_C0_BASELINE", "S2A1_C1_FORCED_REFRESH", "S2A1_C2_LOCAL_CONNECTOR"))
    no_op_fields = ("search_recovery_entry_count", "forced_public_refresh_count", "recovery_plan_active_step_count", "recovery_guidance_changed_step_count", "recovery_effective_intervention_count", "local_connector_attempt_count")
    checks["c0_strict_no_op"] = all(_integer(c0, field) == 0 for field in no_op_fields)
    entries = _integer(c1, "search_recovery_entry_count") + _integer(c2, "search_recovery_entry_count")
    checks["recovery_entries_exist"] = entries > 0
    checks["c2_local_connector_attempt"] = _integer(c2, "local_connector_attempt_count") > 0
    checks["c2_local_connector_plan"] = _integer(c2, "local_connector_plan_count") > 0
    checks["guidance_changed"] = any(_integer(row, "recovery_guidance_changed_step_count") > 0 for row in (c1, c2))
    checks["plan_active"] = any(_integer(row, "recovery_plan_active_step_count") > 0 for row in (c1, c2))
    checks["effective_episode"] = any(_integer(row, "recovery_effective_intervention_episode_count") > 0 for row in (c1, c2))
    checks["single_manifest"] = len({row.get("manifest_sha256") for row in by_variant.values()}) == 1
    checks["v2_schema"] = all(row.get("search_collision_recovery_schema") == "bser.phase1c.prrac.search_collision_recovery.v2" for row in by_variant.values())
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = validate_activation(args.summary_csv)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
