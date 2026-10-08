"""Exact performance evolution; all historical source seals remain frozen."""
from pathlib import Path
from chapter3_bser.experiments.safe_search_v1 import provenance as old

MANIFEST = "docs/provenance/d2_performance_v1_evolution.json"
PARENT = "docs/provenance/d2_suite_pipeline_v1_evolution.json"
REVIEW = "docs/chapter3/d2_suite_v1/performance/source_review.md"
CHANGED = {
    "core/env/target_motion.py",
    "core/env/target_motion.py",
    "core/mapping/path_planner.py",
    "chapter3_bser/experiments/d2_v1/assembly.py",
    "chapter3_bser/experiments/baselines/common/checkpoint.py",
    "chapter3_bser/experiments/baselines/common/train.py",
    "chapter3_bser/experiments/hgr/train.py",
    "chapter3_bser/experiments/d2_suite_v1/plan.py",
    "chapter3_bser/experiments/d2_suite_v1/cli.py",
    "chapter3_bser/experiments/d2_suite_v1/linux.py",
    "chapter3_bser/experiments/safe_search_v1/provenance.py",
}
ADDED = {"chapter3_bser/experiments/d2_performance/" + name + ".py" for name in
         ("__init__", "planning", "logging", "options", "compute", "provenance")}
ADDED |= {"tools/benchmark_d2_performance.py"}


def extend_profile(root, profile, historical):
    root = Path(root)
    manifest, parent = (old.read_json(root / p) for p in (MANIFEST, PARENT))
    if (manifest.get("schema") != "ch3.d2_performance.reviewed_evolution.v1"
            or manifest.get("review_status") != "reviewed"
            or manifest.get("sha256") != old.digest({k: v for k, v in manifest.items() if k != "sha256"})
            or manifest.get("parent_manifest_sha256") != old.file_sha256(root / PARENT)
            or manifest.get("source_review_sha256") != old.file_sha256(root / REVIEW)
            or set(manifest.get("changed_paths", [])) != CHANGED
            or set(manifest.get("added_paths", [])) != ADDED
            or set(manifest.get("profiles", {})) != set(parent["profiles"])):
        raise ValueError("invalid D2 performance reviewed evolution")
    for name, record in manifest["profiles"].items():
        before, expected, changes = parent["profiles"][name]["files"], record["files"], record["changes"]
        old._validate_files(expected)
        if (set(expected) != set(before) | ADDED or set(changes) != CHANGED | ADDED
                or record.get("sha256") != old.digest(expected) or set(before) & ADDED):
            raise ValueError("D2 performance exact inventory mismatch")
        for path, sha in before.items():
            if expected.get(path) != changes.get(path, {}).get("after", sha):
                raise ValueError("unreviewed D2 performance evolution: " + path)
        for path, change in changes.items():
            if change != dict(before=before.get(path), after=expected.get(path)):
                raise ValueError("D2 performance before/after mismatch: " + path)
    if historical != parent["profiles"][profile]["files"]:
        raise ValueError("D2 performance parent inventory mismatch")
    return manifest["profiles"][profile]["files"]
