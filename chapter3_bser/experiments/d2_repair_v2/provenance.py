"""Exact full source inventory, preserving all historical manifests unchanged."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = "docs/provenance/d2_repair_v2_evolution.json"
PARENT_COMMIT = "6538c3e6a67e96d56940ab09fcff15cd9dc7a7d8"


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()


def verify_reviewed_changes(root=ROOT):
    root = Path(root)
    raw = (root/MANIFEST).read_bytes()
    data = json.loads(raw)
    if (data.get("parent_commit") != PARENT_COMMIT or data.get("historical_manifests_modified") is not False
            or data.get("schema") != "ch3.d2.repair_evolution.v2"
            or data.get("sha256") != digest({k:v for k,v in data.items() if k != "sha256"})):
        raise ValueError("invalid independent repair v2 source review")
    inventory = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for package in ("core","chapter3_bser") for p in sorted((root/package).rglob("*.py"))}
    actual = dict(inventory)
    for name in data["nonproduction_files"]:
        actual[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    profiles = data["profiles"]
    matching = [name for name,profile in profiles.items() if profile["files"] == actual]
    if len(matching) != 1:
        differences = {name:sorted(p for p in actual.keys() | profile["files"].keys()
                        if actual.get(p) != profile["files"].get(p)) for name,profile in profiles.items()}
        raise ValueError("repair v2 full source inventory mismatch: "+json.dumps(differences))
    return dict(review_sha256=hashlib.sha256(raw).hexdigest(),profile=matching[0],
                inventory_sha256=digest(inventory),parent_commit=PARENT_COMMIT)


def training_admission(*, mechanism=False, config=None):
    identity = verify_reviewed_changes()
    if mechanism:
        if (config is None or config["max_steps"] > 32 or config["total_main_trajectories"] > 8
                or not 1 <= (config.get("max_total_environment_steps") or 0) <= 256):
            raise ValueError("mechanism execution requires <=32 steps/episode, <=8 episodes and <=256 total start budget")
        return identity
    receipt = ROOT/"docs/chapter3/d2_repair_v2/evidence/acceptance.json"
    if not receipt.is_file():
        raise ValueError("manual pilot admission is not established: missing acceptance evidence")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    required = ("targeted_passed","regression_passed","source_passed","bounded_chain_passed","historical_e0_passed")
    if (value.get("status") != "READY_FOR_MANUAL_PILOT" or value.get("source_identity") != identity
            or any(value.get(k) is not True for k in required)):
        raise ValueError("manual pilot admission rejected; inspect acceptance evidence and unresolved source/E0 gates")
    for name,sha in value.get("evidence_files",{}).items():
        path = (ROOT/name).resolve()
        if ROOT not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("manual pilot admission evidence changed")
    if not value.get("evidence_files"):
        raise ValueError("manual pilot admission has no verified evidence")
    return identity
