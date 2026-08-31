"""Validate the frozen C0 Ep100 Val100 gate without modifying evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from chapter3_bser.experiments.phase1c_prrac.search_collision_recovery import validate_s2a_baseline_regression


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args(argv)
    output=args.output_dir.resolve(); episode_path=output/"search_collision_recovery_episode.csv"; manifest_path=output/"evaluation_manifest.json"
    with episode_path.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
    for row in rows:
        for key in ("found","success"):
            row[key]=str(row.get(key,"")).lower()=="true"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")); expected=[str(row["scenario_id"]) for row in manifest["scenarios"]]
    result=validate_s2a_baseline_regression(rows,expected_scenario_ids=expected)
    print(json.dumps(result,sort_keys=True)); return 0 if result["status"]=="PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
