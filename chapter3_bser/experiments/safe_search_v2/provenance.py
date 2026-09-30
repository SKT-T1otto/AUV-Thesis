"""Exact additive evolution of the frozen SafeSearch v1 source inventory."""
from pathlib import Path
from chapter3_bser.experiments.safe_search_v1 import provenance as old

MANIFEST = "docs/provenance/safe_search_v2_evolution.json"
REVIEW = "docs/chapter3/search_diagnostics/safe_search_v2/source_review.md"
HOOK = "chapter3_bser/experiments/safe_search_v1/provenance.py"


def extend_profile(root, profile, historical):
    root = Path(root)
    manifest = old.read_json(root / MANIFEST)
    if (manifest.get("schema") != "ch3.safe_search.reviewed_evolution.v2"
            or manifest.get("review_status") != "reviewed"
            or manifest.get("sha256") != old.digest({k: v for k, v in manifest.items() if k != "sha256"})
            or manifest.get("parent_manifest_sha256") != old.file_sha256(root / old.MANIFEST)
            or manifest.get("source_review_sha256") != old.file_sha256(root / REVIEW)
            or set(manifest.get("profiles", {})) != {"git_lf", "windows_existing", "git_clone_preserved"}):
        raise ValueError("invalid reviewed SafeSearch v2 evolution")
    additions = manifest.get("added_paths", [])
    if (not additions or len(additions) != len(set(additions)) or any(
            not (n.startswith("chapter3_bser/experiments/safe_search_v2/") and n.endswith(".py"))
            for n in additions)):
        raise ValueError("unexpected SafeSearch v2 addition scope")
    if set(additions) & historical.keys():
        raise ValueError("SafeSearch v2 addition replaced historical source")
    # Validate every profile, including profiles unused by this checkout.
    parent = old.read_json(root / old.MANIFEST)
    for name, record in manifest["profiles"].items():
        before = parent["profiles"][name]["files"]
        changes, expected = record["changes"], record["files"]
        old._validate_files(expected)
        if (set(changes) != {HOOK} | set(additions)
                or record.get("sha256") != old.digest(expected)
                or set(expected) != set(before) | set(additions)):
            raise ValueError("SafeSearch v2 exact inventory mismatch")
        for path, sha in before.items():
            if expected.get(path) != changes.get(path, {}).get("after", sha):
                raise ValueError("unreviewed historical evolution: " + path)
        for path, change in changes.items():
            if change != dict(before=before.get(path), after=expected.get(path)):
                raise ValueError("SafeSearch v2 before/after mismatch: " + path)
    if historical != parent["profiles"][profile]["files"]:
        raise ValueError("SafeSearch v2 parent profile differs")
    return manifest["profiles"][profile]["files"]


def framework_sources():
    result = old.framework_sources()
    if "successor_evolution_sha256" not in result:
        raise ValueError("SafeSearch v2 requires its reviewed source manifest")
    return result


def verify_sources(before):
    if before != framework_sources():
        raise ValueError("SafeSearch v2 source changed during evaluation")
