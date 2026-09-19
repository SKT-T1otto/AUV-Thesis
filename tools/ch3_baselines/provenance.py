"""Separate baseline provenance; historical HGR manifests are never rewritten."""
import hashlib
import json
from pathlib import Path

from chapter3_bser.experiments.hgr.provenance import fresh_source_identity, require_source_match, checkout_identity

ROOT = Path(__file__).resolve().parents[2]
PIN = ROOT / "docs/chapter3/baselines/frozen_production_source.json"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def baseline_source_identity(root=ROOT):
    root = Path(root)
    paths = set((root / "tools/ch3_baselines").rglob("*.py"))
    paths.update((root / "configs/chapter3").rglob("*.json"))
    paths.update(root / name for name in ("scripts/run_ch3_basic_prior_eval.bat", "scripts/linux/run_ch3_basic_prior_eval.sh"))
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(paths)}
    return dict(schema="ch3.basic_search_prior.source.v1", files=files, sha256=digest(files))


def capture_sources():
    production = fresh_source_identity()
    require_source_match(json.loads(PIN.read_text(encoding="utf-8"))["production"], production,
                         context="baseline frozen source gate")
    return dict(production=production, baseline=baseline_source_identity(), pin_sha256=file_sha256(PIN))


def require_unchanged(before, after):
    if before != after:
        raise ValueError("baseline or production source/config inventory changed during evaluation")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
