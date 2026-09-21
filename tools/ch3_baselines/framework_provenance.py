"""Linux baseline provenance with unchanged production and B0 algorithm gates."""
import json
from pathlib import Path

from .provenance import ROOT, capture_sources, digest, file_sha256, require_unchanged

SCRIPT_FILES = (
    "scripts/linux/run_ch3_basic_prior_eval.sh",
    "scripts/linux/run_ch3_baseline_eval.sh",
    "scripts/linux/train_ch3_direct_mc.sh",
    "scripts/linux/train_ch3_direct_boundary.sh",
)
PROTECTED_B0 = ROOT / "docs/chapter3/baselines/framework_protected_b0.json"
PROTECTED_BASELINE = ROOT / "docs/chapter3/baselines/framework_protected_baseline.json"


def baseline_protected_identity(root=ROOT):
    """Scan baseline source/configs and Bash scripts referencing this namespace.

    Hash actual file bytes, including line endings. The reviewed manifest lives
    outside these directories so it can also protect this module without a
    self-referential hash. This inventory is identical on every host: Windows
    launchers are tested separately, never read or hashed here. Linux script
    bytes remain exact; no newline normalization weakens their protection.
    """
    root = Path(root)
    paths = set((root / "tools/ch3_baselines").rglob("*.py"))
    paths.update((root / "configs/chapter3/baselines").rglob("*.json"))
    for path in (root / "scripts").rglob("*.sh"):
        if path.is_file():
            content = path.read_bytes()
            if any(marker in content for marker in (b"tools.ch3_baselines", b"tools/ch3_baselines", b"tools\\ch3_baselines")):
                paths.add(path)
    files = {path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(paths)}
    return dict(files=files, sha256=digest(files))


def framework_sources():
    protected = json.loads(PROTECTED_B0.read_text(encoding="utf-8"))
    for name, expected in protected["files"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"protected B0 implementation changed: {name}")
    result = capture_sources()
    protected = json.loads(PROTECTED_BASELINE.read_text(encoding="utf-8"))
    if (protected.get("schema") != "ch3.baseline_framework.protected_baseline.v1"
            or not isinstance(protected.get("files"), dict)
            or protected.get("sha256") != digest(protected["files"])):
        raise ValueError("invalid protected baseline manifest")
    if protected.get("production_source_sha256") != result["production"]["sha256"]:
        raise ValueError("protected baseline manifest must retain the frozen production source hash")
    current = baseline_protected_identity()
    if current["files"] != protected["files"]:
        changed = sorted(name for name in current["files"].keys() | protected["files"].keys()
                         if current["files"].get(name) != protected["files"].get(name))
        raise ValueError("protected baseline inventory changed; review before refreshing hashes: " + ", ".join(changed))
    result["protected_baseline"] = current
    result["protected_baseline_manifest_sha256"] = file_sha256(PROTECTED_BASELINE)
    scripts = {name: file_sha256(ROOT / name) for name in SCRIPT_FILES}
    result["framework_entry_scripts"] = dict(files=scripts, sha256=digest(scripts))
    result["protected_b0_manifest_sha256"] = file_sha256(PROTECTED_B0)
    return result


def verify_framework_sources(before):
    require_unchanged(before, framework_sources())
