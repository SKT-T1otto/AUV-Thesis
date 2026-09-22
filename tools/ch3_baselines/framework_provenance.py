"""CH3-final production and baseline gates, independent of Windows launchers."""
import json

from .provenance import (
    ROOT, baseline_source_identity, production_sources, digest, file_sha256, require_unchanged, baseline_evolution,
)

SCRIPT_FILES = (
    "scripts/linux/run_ch3_basic_prior_eval.sh",
    "scripts/linux/run_ch3_baseline_eval.sh",
    "scripts/linux/train_ch3_direct_mc.sh",
    "scripts/linux/train_ch3_direct_boundary.sh",
)
PROTECTED_BASELINE = ROOT / "docs/chapter3/baselines/framework_protected_baseline.json"


def baseline_protected_identity(root=None):
    """Scan all baseline Python/config files and all Linux shell scripts.

    Hash actual file bytes, including line endings. The reviewed manifest lives
    outside these directories so it can also protect this module without a
    self-referential hash. This inventory is identical on every host: Windows
    launchers are tested separately, never read or hashed here. Linux script
    bytes remain exact; no newline normalization weakens their protection.
    """
    return baseline_source_identity(root)


def framework_sources():
    result = production_sources()
    protected = json.loads(PROTECTED_BASELINE.read_text(encoding="utf-8"))
    if (protected.get("schema") != "ch3.final_experiment.protected_baseline.v1"
            or not isinstance(protected.get("files"), dict)
            or protected.get("sha256") != digest(protected["files"])):
        raise ValueError("invalid protected baseline manifest")
    if protected.get("production_source_sha256") != result["production_source_sha256"]:
        raise ValueError("protected baseline manifest must match the CH3-final production source hash")
    current = baseline_protected_identity()
    evolution = baseline_evolution()
    if evolution["historical_protected_sha256"] != protected["sha256"]:
        raise ValueError("baseline evolution historical protected binding mismatch")
    expected = dict(protected["files"])
    permitted = {
        "tools/ch3_baselines/registry.py", "tools/ch3_baselines/run_training.py",
        "tools/ch3_baselines/run_baseline.py", "tools/ch3_baselines/learned_evaluation.py",
        "tools/ch3_baselines/provenance.py", "tools/ch3_baselines/framework_provenance.py",
        "configs/chapter3/baselines/baseline_registry.json",
        "configs/chapter3/baselines/direct_mc_train.json", "configs/chapter3/baselines/direct_boundary_train.json",
    }
    if set(evolution["protected_changes"]) != permitted:
        raise ValueError("independent baseline evolution has unexpected protected paths")
    for name, record in evolution["protected_changes"].items():
        if record["before"] != expected.get(name):
            raise ValueError("baseline evolution does not preserve historical protected hashes")
        expected[name] = record["after"]
    if current["files"] != expected:
        changed = sorted(name for name in current["files"].keys() | expected.keys()
                         if current["files"].get(name) != expected.get(name))
        raise ValueError("protected baseline inventory changed; review before refreshing hashes: " + ", ".join(changed))
    result["protected_baseline"] = current
    result["baseline"] = current
    scripts = {name: file_sha256(ROOT / name) for name in SCRIPT_FILES}
    result["framework_entry_scripts"] = dict(files=scripts, sha256=digest(scripts))
    return result


def verify_framework_sources(before):
    require_unchanged(before, framework_sources())
