"""Exact repair evolution, independent of historical experiment source gates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "docs/provenance/d2_repair_v1_evolution.json"


def verify_reviewed_changes():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data["historical_manifests_modified"] is not False:
        raise ValueError("historical manifests must remain frozen")
    expected = {r["path"]: r["current_sha256"] for r in data["records"]}
    expected.update(data["new_files"])
    for name, value in expected.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value:
            raise ValueError(f"unreviewed repair source drift: {name}")
    package = ROOT / "chapter3_bser/experiments/d2_repair_v1"
    actual = {p.relative_to(ROOT).as_posix() for p in package.rglob("*.py")}
    pinned = {p for p in data["new_files"] if p.startswith("chapter3_bser/experiments/d2_repair_v1/")}
    if actual != pinned:
        raise ValueError("repair package inventory mismatch")
    return data
