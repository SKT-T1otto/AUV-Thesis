"""Separate baseline provenance; historical HGR manifests are never rewritten."""
import hashlib
import json
from pathlib import Path

from chapter3_bser.experiments.hgr.provenance import (
    fresh_source_identity, require_source_match, validate_source_identity, checkout_identity,
)

ROOT = Path(__file__).resolve().parents[2]
PIN = ROOT / "docs/chapter3/baselines/final_production_source.json"
EVOLUTION_RELATIVE = "docs/provenance/baseline_maddpg_evolution.json"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def baseline_source_identity(root=None):
    root = ROOT if root is None else Path(root)
    paths = set((root / "tools/ch3_baselines").rglob("*.py"))
    paths.update((root / "configs/chapter3/baselines").rglob("*.json"))
    paths.update((root / "scripts").rglob("*.sh"))
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(paths)}
    return dict(files=files, sha256=digest(files))


def baseline_evolution():
    """Explicit 2026-09-22 addition; historical production pins stay frozen."""
    value = json.loads((ROOT / EVOLUTION_RELATIVE).read_text(encoding="utf-8"))
    if (value.get("schema") != "ch3.baseline.maddpg_evolution.v1"
            or value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"})):
        raise ValueError("invalid independent baseline evolution manifest")
    added = value.get("added_production", {})
    if not added or any(not name.startswith("chapter3_bser/experiments/baselines/") or not name.endswith(".py") for name in added):
        raise ValueError("baseline evolution may only add independent training sources")
    return value


def production_sources():
    """Check exact reviewed checkout bytes without changing checkpoint identity.

    The Git/LF inventory names the final experiment. The existing Windows
    working tree has six precisely recorded CRLF/mixed files. Only these two
    complete, reviewed inventories are accepted; no bytes are normalized here.
    HGR's native raw-byte identity is retained for its checkpoint comparisons.
    """
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    if pin.get("schema") != "ch3.final_experiment.production.v1":
        raise ValueError("invalid CH3-final production manifest")
    final = validate_source_identity(pin["production"])
    evolution = baseline_evolution()
    if evolution["historical_production_sha256"] != final["sha256"]:
        raise ValueError("baseline evolution historical production binding mismatch")
    production = fresh_source_identity()
    if set(pin["reviewed_worktree_profiles"]) != {"windows_existing"}:
        raise ValueError("invalid reviewed CH3-final worktree profiles")
    profiles = {"git_lf": final, **pin["reviewed_worktree_profiles"]}
    for expected in profiles.values():
        validate_source_identity(expected)
    for name, expected in profiles.items():
        if set(expected["files"]) & set(evolution["added_production"]):
            raise ValueError("baseline evolution cannot replace a historical production record")
        evolved_files = {**expected["files"], **evolution["added_production"]}
        evolved = dict(expected, files=evolved_files,
                       sha256=hashlib.sha256(json.dumps(evolved_files, sort_keys=True).encode()).hexdigest())
        validate_source_identity(evolved)
        if production == evolved:
            return dict(experiment="ch3.final_experiment.v1",
                        production_source_sha256=final["sha256"],
                        production_checkout_profile=name, production=production,
                        baseline_evolution_sha256=evolution["sha256"])
    require_source_match(evolved, production, context="CH3-final source gate plus independent baselines")


def capture_sources():
    # Both the original B0 entry and the unified entry enforce the complete
    # current gate. Historical manifests are retained but are never loaded.
    from .framework_provenance import framework_sources
    return framework_sources()


def require_unchanged(before, after):
    if before != after:
        raise ValueError("baseline or production source/config inventory changed during evaluation")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
