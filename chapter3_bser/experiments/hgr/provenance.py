"""Production source identity shared by policy checkpoints and simulator state."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

IMPLEMENTATION_VERSION = "hgr.exact_score_cycle.v1"


@lru_cache(maxsize=1)
def source_identity():
    root = Path(__file__).resolve().parents[3]
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for package in ("core", "chapter3_bser") for p in sorted((root/package).rglob("*.py"))}
    return dict(implementation_version=IMPLEMENTATION_VERSION,
                sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files=files)
