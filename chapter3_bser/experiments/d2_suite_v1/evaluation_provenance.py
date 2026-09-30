"""Exact evaluation-entry review; the previous Linux launch seal stays frozen."""
from pathlib import Path
from chapter3_bser.experiments.safe_search_v1 import provenance as old

MANIFEST = "docs/provenance/d2_suite_linux_evaluation_v1_evolution.json"
PARENT = "docs/provenance/d2_suite_linux_v1_evolution.json"
REVIEW = "docs/chapter3/d2_suite_v1/linux_evaluation/source_review.md"
CHANGED = {"chapter3_bser/experiments/safe_search_v1/provenance.py"}
ADDED = {"chapter3_bser/experiments/d2_suite_v1/evaluation_cli.py",
         "chapter3_bser/experiments/d2_suite_v1/evaluation_provenance.py",
         "scripts/linux/evaluate_d2_reference.sh", "scripts/linux/evaluate_d2_trained.sh",
         "scripts/linux/summarize_d2_suite.sh"}


def extend_profile(root, profile, historical):
    root = Path(root)
    manifest, parent = (old.read_json(root / p) for p in (MANIFEST, PARENT))
    if (manifest.get("schema") != "ch3.d2_suite_linux_evaluation.reviewed_evolution.v1"
            or manifest.get("review_status") != "reviewed"
            or manifest.get("sha256") != old.digest({k: v for k, v in manifest.items() if k != "sha256"})
            or manifest.get("parent_manifest_sha256") != old.file_sha256(root / PARENT)
            or manifest.get("source_review_sha256") != old.file_sha256(root / REVIEW)
            or set(manifest.get("changed_paths", [])) != CHANGED
            or set(manifest.get("added_paths", [])) != ADDED
            or set(manifest.get("profiles", {})) != set(parent["profiles"])):
        raise ValueError("invalid D2 Linux evaluation reviewed evolution")
    for name, record in manifest["profiles"].items():
        before, expected, changes = parent["profiles"][name]["files"], record["files"], record["changes"]
        old._validate_files(expected)
        if (set(expected) != set(before) | ADDED or set(changes) != CHANGED | ADDED
                or record.get("sha256") != old.digest(expected) or set(before) & ADDED):
            raise ValueError("D2 Linux evaluation exact inventory mismatch")
        for path, sha in before.items():
            if expected.get(path) != changes.get(path, {}).get("after", sha):
                raise ValueError("unreviewed D2 Linux evaluation evolution: " + path)
        for path, change in changes.items():
            if change != dict(before=before.get(path), after=expected.get(path)):
                raise ValueError("D2 Linux evaluation before/after mismatch: " + path)
    if historical != parent["profiles"][profile]["files"]:
        raise ValueError("D2 Linux evaluation parent inventory mismatch")
    return manifest["profiles"][profile]["files"]
