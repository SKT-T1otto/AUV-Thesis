"""Independent, read-only verification of the exact v2 source review."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENT="6538c3e6a67e96d56940ab09fcff15cd9dc7a7d8"
BEFORE={"core/algorithms/maddpg.py":"fbf983ec992613a45c10a2ced45725c17b2a412414d64167cb02b33ab8692a91",
        "core/replay/ch3_buffer.py":"e22412e43e589962582c2a3ce727b167a6689e60223f6d64fd71ef1e9e5cfa42"}

def verify(root=ROOT):
    root=Path(root)
    data=json.loads((root/"docs/provenance/d2_repair_v2_evolution.json").read_text(encoding="utf-8"))
    canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
    assert data["parent_commit"]==PARENT and data["historical_manifests_modified"] is False
    assert data["sha256"]==canonical({k:v for k,v in data.items() if k!="sha256"})
    records={r["path"]:r for r in data["records"]}
    assert set(records)==set(BEFORE)
    for path,old in BEFORE.items():
        r=records[path]; assert r["previous_reviewed_sha256"]==old and r["reason"]
        raw=(root/path).read_bytes(); assert hashlib.sha256(raw).hexdigest()==r["current_sha256"]
        dump=ast.dump(ast.parse(raw.decode()),annotate_fields=True,include_attributes=True)
        assert hashlib.sha256(dump.encode()).hexdigest()==r["current_ast_dump_sha256"]
    actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for package in ("core","chapter3_bser") for p in (root/package).rglob("*.py")}
    for name in data["nonproduction_files"]:
        actual[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    matched=[n for n,p in data["profiles"].items() if p["files"]==actual]
    assert len(matched)==1,"complete production/config/historical-manifest inventory mismatch"
    profile=data["profiles"][matched[0]]
    assert set(profile["parent_files"]) | set(data["added_files"]) == {p for p in actual if p.endswith('.py') and p.startswith(('core/','chapter3_bser/'))}
    assert set(data["unchanged_files"]) == set(profile["parent_files"])-set(records)
    for name in data["unchanged_files"]: assert actual[name]==profile["parent_files"][name]
    historical=json.loads((root/"docs/provenance/ch3_to_core_migration_manifest.json").read_text(encoding="utf-8"))
    assert len(historical["records"])==27
    assert all((root/r["new_core_path"]).is_file() for r in historical["records"])
    return dict(source_passed=True,profile=matched[0],parent_commit=PARENT,modified=list(records),
                added=list(data["added_files"]),unchanged=len(data["unchanged_files"]))

if __name__=="__main__": print(json.dumps(verify(),indent=2))
