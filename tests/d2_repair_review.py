"""Exact successor checks; historical hashes stay in their original assertions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT/"docs/provenance/d2_repair_v1_evolution.json").read_text())
assert MANIFEST["historical_manifests_modified"] is False
RECORDS = {r["path"]:r for r in MANIFEST["records"]}
assert set(RECORDS) == {"core/algorithms/maddpg.py", "core/env/uav_env.py",
                       "chapter3_bser/experiments/baselines/common/runtime.py"}
# Pin the reviewed parent bytes independently of the new manifest.
assert {p:r["previous_reviewed_sha256"] for p,r in RECORDS.items()} == {
    "core/algorithms/maddpg.py":"a574285b33d9a82971cf8016ce93e79f1c90b07cca5236edf5deb7fcb9b6d907",
    "core/env/uav_env.py":"2626d3f957a8e448db34868442d28cc20988cbb46292dd48700eab0335ad7eaf",
    "chapter3_bser/experiments/baselines/common/runtime.py":"a9c8030d9e146c50f5ed76efa60645957661072ef566da67082cd481753cb56b",
}
assert {p:r["current_sha256"] for p,r in RECORDS.items()} == {
    "core/algorithms/maddpg.py":"fbf983ec992613a45c10a2ced45725c17b2a412414d64167cb02b33ab8692a91",
    "core/env/uav_env.py":"402878207aa69f334311e1c690dbc7b231a7fde0a0a7affaf5c645c1dd24d60b",
    "chapter3_bser/experiments/baselines/common/runtime.py":"d494fcee4f9c73368557999f4ac9e5736d3418ac9bcf30e45351c65e5fc226a6",
}


def successor(path, expected):
    record = RECORDS.get(path)
    if record is None:
        return expected
    assert record["previous_reviewed_sha256"] == expected, path
    return record["current_sha256"]


def add_evolutions(mapping, historical_key="historical_sha256"):
    historical = json.loads((ROOT/"docs/provenance/ch3_to_core_migration_manifest.json").read_text())
    original = {r["new_core_path"]:r["new_core_sha256"] for r in historical["records"]}
    for path,record in RECORDS.items():
        if path not in original:
            continue
        previous = mapping.get(path)
        if previous is not None:
            assert previous[historical_key] == original[path]
            assert previous["current_sha256"] == record["previous_reviewed_sha256"]
        else:
            assert original[path] == record["previous_reviewed_sha256"]
        mapping[path] = {historical_key:original[path],"current_sha256":record["current_sha256"]}
