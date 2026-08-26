"""Verify the Phase-1A pre-existing core freeze and emit a reproducible report."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEFORE = ROOT / "docs" / "chapter3_bser" / "phase1a" / "core_freeze_before.json"


def build_after(before_path=DEFAULT_BEFORE):
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    records = []
    changed = []
    for expected in before["files"]:
        path = ROOT / expected["path"]
        data = path.read_bytes()
        dump = ast.dump(ast.parse(data.decode("utf-8")), annotate_fields=True, include_attributes=True)
        record = {
            "path": expected["path"],
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "ast_dump_sha256": hashlib.sha256(dump.encode("utf-8")).hexdigest(),
        }
        records.append(record)
        if record != {key: expected[key] for key in record}:
            changed.append(expected["path"])
    return {
        "schema": "bser.phase1a.core_freeze_after.v1",
        "base_head": before["base_head"],
        "existing_core_python_count": len(records),
        "existing_core_python_files_changed": len(changed),
        "changed_paths": changed,
        "new_core_python_paths": [
            "core/mapping/planning_state.py",
            "core/mapping/travel_cost_service.py",
        ],
        "files": records,
        "passed": not changed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_after(args.before)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
