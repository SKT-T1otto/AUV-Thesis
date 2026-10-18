"""New opt-in gate: preserve every parent byte and reject unreviewed additions."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PARENT='docs/provenance/d2_performance_v2_evolution.json'
SEAL='docs/provenance/long_horizon_maddpg_v1_evolution.json'
ADDED={f'chapter3_bser/experiments/long_horizon_maddpg_v1/{name}.py' for name in
       ('__init__','config','model','runtime','provenance','cli')}
ADDED |= {'configs/chapter3/long_horizon_maddpg_v1.json','tests/test_long_horizon_maddpg_v1.py',
          'docs/chapter3/long_horizon_maddpg_v1/README.md'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def preflight(root=ROOT):
    from chapter3_bser.experiments.safe_search_v1.provenance import inventory
    root=Path(root)
    parent=json.loads((root/PARENT).read_text(encoding='utf-8'))
    seal=json.loads((root/SEAL).read_text(encoding='utf-8'))
    if (seal.get('schema') != 'ch3.long_horizon_maddpg.reviewed_evolution.v1'
            or seal.get('sha256') != digest({k:v for k,v in seal.items() if k!='sha256'})
            or seal.get('parent_sha256') != sha(root/PARENT)
            or set(seal.get('added',{})) != ADDED
            or any(sha(root/name)!=value for name,value in seal['added'].items())):
        raise ValueError('long-horizon source seal mismatch')
    for profile,record in parent['profiles'].items():
        expected=record['files']
        if all((root/name).is_file() and sha(root/name)==value for name,value in expected.items()):
            current=inventory(root)['files']
            # Tests/docs are sealed separately and are outside the production glob.
            permitted=set(expected)|ADDED
            extras=sorted(set(current)-permitted)
            identity=dict(files=current,parent_profile=profile,experiment_files=seal['added'],seal_sha256=sha(root/SEAL))
            identity['sha256']=digest(identity)
            return dict(ready=not extras,unreviewed_additions=extras,source=identity,
                        historical_gates_modified=False)
    raise ValueError('no complete frozen parent source profile matches; historical bytes changed')


def require_ready():
    status=preflight()
    if not status['ready']:
        raise ValueError('unreviewed source additions block execution: '+', '.join(status['unreviewed_additions']))
    return status['source']
