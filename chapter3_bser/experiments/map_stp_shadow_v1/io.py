"""JSON-only public snapshot transport and exact experiment source identity."""
from dataclasses import asdict, is_dataclass
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from core.mapping.planning_graph import (
    EndpointConnectorSet, PlanningConnectorView, PlanningEdgeView, PlanningGraphView,
)
from core.mapping.planning_state import (
    GridGeometryView, OccupancyBeliefView, PlanningAgentView, PlanningStateView, TargetBeliefView,
)

ROOT = Path(__file__).resolve().parents[3]


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [plain(v) for v in value]
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def new_directory(path, prefix):
    path, parent = Path(path).resolve(), (ROOT/prefix).resolve()
    if parent not in path.parents or path.exists():
        raise ValueError('a new subdirectory is required below '+str(parent))
    path.mkdir(parents=True)
    return path


def source_identity():
    files = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
             for directory, pattern in [('core','*.py'),('chapter3_bser','*.py'),
                 ('tools/ch3_baselines','*.py'),('configs','*.json'),('scripts','*.py'),('scripts','*.sh')]
             for p in (ROOT/directory).rglob(pattern)}
    git = lambda *args: subprocess.check_output(['git','-c','core.fsmonitor=false',*args],cwd=ROOT,text=True).strip()
    return dict(head=git('rev-parse','HEAD'), branch=git('branch','--show-current'),
                working_tree=git('status','--short'), files=files, source_sha256=digest(files))


def locked(value, dtype=float):
    a = np.asarray(value, dtype=dtype).copy(); a.setflags(write=False)
    return a


def state_from_dict(data):
    d = dict(data)
    g = dict(d['grid'])
    for name in ('shape','origin','spacing'): g[name] = tuple(g[name])
    g['cell_centers'] = locked(g['cell_centers']); d['grid'] = GridGeometryView(**g)
    b = dict(d['target_belief']); b['probabilities'] = locked(b['probabilities'])
    d['target_belief'] = TargetBeliefView(**b)
    o = dict(d['occupancy'])
    for key in ('occupancy_probability','known_mask','free_mask','occupied_mask','unknown_mask'):
        o[key] = locked(o[key], float if key == 'occupancy_probability' else bool)
    d['occupancy'] = OccupancyBeliefView(**o)
    agents = []
    for agent in d['agents']:
        a = dict(agent)
        for key in ('position','velocity','current_navigation_target'):
            if a[key] is not None: a[key] = tuple(a[key])
        agents.append(PlanningAgentView(**a))
    d['agents'] = tuple(agents); d['searcher_ids'] = tuple(d['searcher_ids'])
    g = dict(d['planning_graph']); g['shape'] = tuple(g['shape'])
    for key,dtype in [('cell_centers',float),('valid_mask',bool),('component_labels',int)]:
        g[key] = locked(g[key],dtype)
    for key in ('searcher_adjacency','executor_adjacency'):
        g[key] = tuple(tuple(PlanningEdgeView(**edge) for edge in row) for row in g[key])
    endpoints = []
    for endpoint in g['endpoint_connectors']:
        e = dict(endpoint); e['point'] = tuple(e['point'])
        e['connectors'] = tuple(PlanningConnectorView(**c) for c in e['connectors'])
        endpoints.append(EndpointConnectorSet(**e))
    g['endpoint_connectors'] = tuple(endpoints)
    for key in ('searcher_vertical_cost_lookup','executor_vertical_cost_lookup'):
        g[key] = tuple(tuple(v) for v in g[key])
    d['planning_graph'] = PlanningGraphView(**g)
    return PlanningStateView(**d)


SNAPSHOT_FIELDS = {'schema','identity','state','public','goals','vehicles','state_sha256'}


def save_snapshot(path, payload):
    if set(payload) != SNAPSHOT_FIELDS:
        raise ValueError('unexpected public snapshot fields')
    data = json.dumps(plain(payload),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    # Deterministic gzip bytes, no pickle or executable object serialization.
    Path(path).write_bytes(gzip.compress(data,mtime=0))


def load_snapshot(path):
    payload = json.loads(gzip.decompress(Path(path).read_bytes()))
    if set(payload) != SNAPSHOT_FIELDS or payload['schema'] != 'ch3.map_stp.public_snapshot.v1':
        raise ValueError('unexpected public snapshot schema/fields')
    from core.mapping.planning_state import planning_state_sha256
    state = state_from_dict(payload['state'])
    if planning_state_sha256(state) != payload['state_sha256']:
        raise ValueError('public state identity mismatch')
    payload['state'] = state
    return payload
