"""Additional framework provenance; neither HGR nor B0 provenance is modified."""
import json

from .provenance import ROOT, capture_sources, digest, file_sha256, require_unchanged

SCRIPT_FILES = (
    "scripts/run_ch3_baseline_eval.bat", "scripts/linux/run_ch3_baseline_eval.sh",
    "scripts/train_ch3_direct_mc.bat", "scripts/linux/train_ch3_direct_mc.sh",
    "scripts/train_ch3_direct_boundary.bat", "scripts/linux/train_ch3_direct_boundary.sh",
)
PROTECTED_B0 = ROOT / "docs/chapter3/baselines/framework_protected_b0.json"


def framework_sources():
    protected = json.loads(PROTECTED_B0.read_text(encoding="utf-8"))
    for name, expected in protected["files"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"protected B0 implementation changed: {name}")
    result = capture_sources()
    scripts = {name: file_sha256(ROOT / name) for name in SCRIPT_FILES}
    result["framework_entry_scripts"] = dict(files=scripts, sha256=digest(scripts))
    result["protected_b0_manifest_sha256"] = file_sha256(PROTECTED_B0)
    return result


def verify_framework_sources(before):
    require_unchanged(before, framework_sources())
