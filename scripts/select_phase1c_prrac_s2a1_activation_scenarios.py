"""Select deterministic diagnostic-only S2-A.1 scenarios from a v1 C0 episode CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def select_scenarios(baseline_episode_csv: Path, count: int) -> dict[str, object]:
    if int(count) <= 0:
        raise ValueError("count must be positive")
    with Path(baseline_episode_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baseline = [row for row in rows if row.get("search_recovery_variant") == "S2A_C0_BASELINE"]
    if not baseline:
        raise ValueError("baseline episode CSV contains no S2A_C0_BASELINE rows")
    truthy = {"1", "true", "yes"}
    selected = [
        str(row["scenario_id"])
        for row in baseline
        if str(row.get("searcher_collision_episode_pre_found", "")).strip().lower() in truthy
    ][: int(count)]
    if not selected:
        raise ValueError("baseline episode CSV contains no pre-found collision scenarios")
    return {
        "schema": "bser.phase1c.prrac.s2a1.activation_scenario_selection.v1",
        "diagnostic_only": True,
        "scenario_selection_mode": "baseline_collision_targeted_smoke",
        "requested_count": int(count),
        "selected_count": len(selected),
        "scenario_ids": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-episode-csv", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = select_scenarios(args.baseline_episode_csv, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
