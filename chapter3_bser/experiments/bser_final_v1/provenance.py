"""Exact, additive F-experiment source evolution; all earlier seals stay frozen."""
from pathlib import Path
from chapter3_bser.experiments.safe_search_v1 import provenance as old

MANIFEST = "docs/provenance/bser_final_v1_evolution.json"
PARENT = "docs/provenance/bser_effect_v1_evolution.json"
REVIEW = "docs/chapter3/search_diagnostics/bser_final_v1/source_review.md"
HOOK = "chapter3_bser/experiments/safe_search_v1/provenance.py"


def extend_profile(root, profile, historical):
    root = Path(root)
    manifest = old.read_json(root / MANIFEST)
    if (manifest.get("schema") != "ch3.bser_final.reviewed_evolution.v1"
            or manifest.get("review_status") != "reviewed"
            or manifest.get("sha256") != old.digest({k:v for k,v in manifest.items() if k != "sha256"})
            or manifest.get("parent_manifest_sha256") != old.file_sha256(root / PARENT)
            or manifest.get("source_review_sha256") != old.file_sha256(root / REVIEW)
            or set(manifest.get("profiles", {})) != {"git_lf", "windows_existing", "git_clone_preserved"}):
        raise ValueError("invalid BSER F reviewed evolution")
    additions = manifest.get("added_paths", [])
    if (not additions or len(set(additions)) != len(additions) or any(
            not (p.startswith("chapter3_bser/experiments/bser_final_v1/") and p.endswith(".py")) for p in additions)
            or set(additions) & historical.keys()):
        raise ValueError("unexpected BSER F addition scope")
    parent = old.read_json(root / PARENT)
    for name, record in manifest["profiles"].items():
        before = parent["profiles"][name]["files"]
        changes, expected = record["changes"], record["files"]
        old._validate_files(expected)
        if (set(changes) != {HOOK} | set(additions) or record.get("sha256") != old.digest(expected)
                or set(expected) != set(before) | set(additions)):
            raise ValueError("BSER F exact inventory mismatch")
        for path, sha in before.items():
            if expected.get(path) != changes.get(path, {}).get("after", sha):
                raise ValueError("unreviewed BSER historical evolution: " + path)
        for path, change in changes.items():
            if change != dict(before=before.get(path), after=expected.get(path)):
                raise ValueError("BSER F before/after mismatch: " + path)
    if historical != parent["profiles"][profile]["files"]:
        raise ValueError("BSER F parent profile differs")
    return manifest["profiles"][profile]["files"]


def framework_sources():
    result = old.framework_sources()
    if "bser_final_evolution_sha256" not in result:
        raise ValueError("BSER F requires its reviewed source manifest")
    return result


def verify_sources(before):
    if before != framework_sources():
        raise ValueError("BSER F source changed during evaluation")
