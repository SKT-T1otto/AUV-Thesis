"""Explicit local release operation, never invoked by an experiment or test.

Only the F namespace and the reviewed dispatcher can extend the frozen D seal.
Run after source review; this does not rewrite any historical manifest.
"""
from pathlib import Path
import hashlib
import json

from chapter3_bser.experiments.safe_search_v1 import provenance as old
from chapter3_bser.experiments.bser_final_v1.provenance import MANIFEST, PARENT, REVIEW, HOOK


def seal():
    root=old.ROOT
    parent=old.read_json(root/PARENT)
    additions=sorted(p.relative_to(root).as_posix() for p in (root/"chapter3_bser/experiments/bser_final_v1").glob("*.py"))
    current=old.inventory()["files"]
    before=parent["profiles"]["windows_existing"]["files"]
    if set(current)!=set(before)|set(additions):
        raise ValueError("unreviewed addition/removal outside F namespace")
    differences={p for p in before if before[p]!=current[p]}
    if differences!={HOOK}:
        raise ValueError("unreviewed changes to pre-existing source: "+repr(differences))
    profiles={}
    for name,record in parent["profiles"].items():
        expected=dict(record["files"])
        for path in [HOOK,*additions]:
            data=(root/path).read_bytes()
            if name!="windows_existing":data=data.replace(b"\r\n",b"\n")
            expected[path]=hashlib.sha256(data).hexdigest()
        changes={p:dict(before=record["files"].get(p),after=expected[p]) for p in sorted([HOOK,*additions])}
        profiles[name]=dict(files=expected,sha256=old.digest(expected),changes=changes)
    result=dict(schema="ch3.bser_final.reviewed_evolution.v1",review_status="reviewed",
        authorization="2026-09-29 user request: implement F0--F6 experiment; formal runs remain manual",
        parent_manifest_sha256=old.file_sha256(root/PARENT),source_review_sha256=old.file_sha256(root/REVIEW),
        added_paths=additions,profiles=profiles)
    result["sha256"]=old.digest(result)
    (root/MANIFEST).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    verified=old.framework_sources()
    print(json.dumps(dict(checkout_profile=verified["checkout_profile"],inventory=verified["inventory"]["sha256"],
        source_count=len(verified["inventory"]["files"]),historical_records=verified["historical_record_count"])))


if __name__=="__main__":
    seal()
