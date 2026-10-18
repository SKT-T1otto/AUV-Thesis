"""Exact v2 successor layered after the frozen v1 review; no exemptions."""
import json
from pathlib import Path
from tests import d2_repair_review as parent
ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'docs/provenance/d2_repair_v2_evolution.json').read_text(encoding='utf-8'))
V2={r['path']:r for r in DATA['records']}
assert DATA['historical_manifests_modified'] is False
assert {p:r['previous_reviewed_sha256'] for p,r in V2.items()} == {'core/algorithms/maddpg.py': 'fbf983ec992613a45c10a2ced45725c17b2a412414d64167cb02b33ab8692a91', 'core/replay/ch3_buffer.py': 'e22412e43e589962582c2a3ce727b167a6689e60223f6d64fd71ef1e9e5cfa42'}
assert {p:r['current_sha256'] for p,r in V2.items()} == {'core/algorithms/maddpg.py': '206c4b134ac16cfc903cdeabd6ee77de0ca5205dab36702abbf3bdd7ab40ef66', 'core/replay/ch3_buffer.py': '1ad6547e89c8c0291a3633f9a7b46f7c8a3b428577a6dee34115ba2bf04876cb'}
RECORDS=dict(parent.RECORDS,**V2)

def successor(path, expected):
    value=parent.successor(path,expected)
    if path in V2:
        assert V2[path]['previous_reviewed_sha256']==value,path
        value=V2[path]['current_sha256']
    return value

def add_evolutions(mapping,historical_key='historical_sha256'):
    parent.add_evolutions(mapping,historical_key)
    original={r['new_core_path']:r['new_core_sha256'] for r in json.loads((ROOT/'docs/provenance/ch3_to_core_migration_manifest.json').read_text(encoding='utf-8'))['records']}
    for path,record in V2.items():
        if path not in original: continue
        previous=mapping.get(path)
        if previous is not None:
            assert previous[historical_key]==original[path]
            assert previous['current_sha256']==record['previous_reviewed_sha256']
        else: assert original[path]==record['previous_reviewed_sha256']
        mapping[path]={historical_key:original[path],'current_sha256':record['current_sha256']}
