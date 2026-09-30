"""Exact additive D2 integration review; historical seals remain authoritative."""
from pathlib import Path
from chapter3_bser.experiments.safe_search_v1 import provenance as old

MANIFEST = "docs/provenance/d2_v1_evolution.json"
PARENT = "docs/provenance/bser_final_v1_evolution.json"
REVIEW = "docs/chapter3/d2_v1/source_review.md"
REFERENCE = "docs/chapter3/d2_v1/frozen_reference.json"
CHANGED = {
    ".gitattributes",
    ".gitattributes",
    "chapter3_bser/experiments/hgr/runtime.py",
    "chapter3_bser/experiments/hgr/train.py",
    "chapter3_bser/experiments/baselines/common/runtime.py",
    "chapter3_bser/experiments/baselines/common/checkpoint.py",
    "chapter3_bser/experiments/baselines/common/train.py",
    "chapter3_bser/experiments/safe_search_v1/provenance.py",
    "tools/ch3_baselines/registry.py",
    "tools/ch3_baselines/run_baseline.py",
    "tools/ch3_baselines/framework_provenance.py",
}
ADDED = {"chapter3_bser/experiments/d2_v1/" + name + ".py" for name in
         ("__init__", "contract", "assembly", "controller", "provenance", "preflight")}
ADDED |= {"configs/chapter3/d2_v1/" + name + ".json" for name in
          ("hgr_train", "direct_mc_train", "direct_boundary_train", "baseline_registry")}


def extend_profile(root, profile, historical):
    root = Path(root)
    manifest, parent = (old.read_json(root / p) for p in (MANIFEST, PARENT))
    if (manifest.get("schema") != "ch3.d2.reviewed_evolution.v1"
            or manifest.get("review_status") != "reviewed"
            or manifest.get("sha256") != old.digest({k:v for k,v in manifest.items() if k != "sha256"})
            or manifest.get("parent_manifest_sha256") != old.file_sha256(root / PARENT)
            or manifest.get("source_review_sha256") != old.file_sha256(root / REVIEW)
            or manifest.get("frozen_reference_sha256") != old.file_sha256(root / REFERENCE)
            or set(manifest.get("changed_paths", [])) != CHANGED
            or set(manifest.get("added_paths", [])) != ADDED
            or set(manifest.get("profiles", {})) != set(parent["profiles"])):
        raise ValueError("invalid D2 reviewed evolution")
    frozen = old.read_json(root / REFERENCE)
    if frozen.get("parent_manifest_sha256") != old.file_sha256(root / PARENT):
        raise ValueError("D2 reference parent changed")
    # Git converts the four historical result JSONs from CRLF to LF. Accept
    # either complete, explicitly recorded reference profile, never normalize
    # runtime bytes or mix arbitrary per-file alternatives.
    native, git = frozen["frozen_files"], frozen["frozen_git_lf_files"]
    old._validate_files(native)
    old._validate_files(git)
    if set(native) != set(git):
        raise ValueError("D2 reference profile file inventory mismatch")
    actual = {path: old.file_sha256(root / path) for path in native}
    if actual != native and actual != git:
        raise ValueError("frozen D2 reference changed or mixed reference profiles")
    for name, record in manifest["profiles"].items():
        before = parent["profiles"][name]["files"]
        expected, changes = record["files"], record["changes"]
        old._validate_files(expected)
        if (set(expected) != set(before) | ADDED or set(changes) != CHANGED | ADDED
                or record.get("sha256") != old.digest(expected) or set(before) & ADDED):
            raise ValueError("D2 exact inventory mismatch")
        for path, sha in before.items():
            if expected.get(path) != changes.get(path, {}).get("after", sha):
                raise ValueError("unreviewed D2 evolution: " + path)
        for path, record_change in changes.items():
            if record_change != dict(before=before.get(path), after=expected.get(path)):
                raise ValueError("D2 before/after mismatch: " + path)
    if historical != parent["profiles"][profile]["files"]:
        raise ValueError("D2 parent inventory mismatch")
    return manifest["profiles"][profile]["files"]


def framework_sources():
    result = old.framework_sources()
    if "d2_evolution_sha256" not in result:
        raise ValueError("D2 requires its reviewed source seal")
    return result


def verify_if_enabled(config):
    from .contract import enabled
    return framework_sources() if enabled(config) else None


def baseline_sources():
    from chapter3_bser.experiments.hgr.provenance import fresh_source_identity
    from tools.ch3_baselines.provenance import baseline_source_identity
    from tools.ch3_baselines.framework_provenance import SCRIPT_FILES
    result = framework_sources()
    result.update(experiment="ch3.d2.v1", production=fresh_source_identity(),
                  production_source_sha256=result["inventory"]["sha256"],
                  production_checkout_profile=result["checkout_profile"])
    result["baseline"] = result["protected_baseline"] = baseline_source_identity()
    scripts = {p: old.file_sha256(old.ROOT / p) for p in SCRIPT_FILES}
    result["framework_entry_scripts"] = dict(files=scripts, sha256=old.digest(scripts))
    return result
