import copy
import json
import os
from pathlib import Path
import tempfile
from functools import lru_cache
import torch
from chapter3_bser.experiments.d2_repair_v2.replay import ExecutionSample
from chapter3_bser.experiments.baselines.common.checkpoint import state_digest

ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def evidence_dir():
    """Each invocation gets fresh evidence; never overwrite a prior run."""
    configured=os.environ.get("AUV_REPAIR_TEST_EVIDENCE")
    path=Path(configured).resolve() if configured else Path(tempfile.mkdtemp(prefix="auv_d2_repair_tests_"))
    path.mkdir(parents=True,exist_ok=True)
    return path

def config(label="b2"):
    return json.loads((ROOT/f"configs/chapter3/d2_repair_v2/{label}_train.json").read_text(encoding="utf-8"))

def sample(n=8, frozen=False, terminal=False):
    obs=torch.randn(n,4,28)
    obs[:,:,20]=0
    obs[:,:,22:26]=torch.eye(4)
    obs[:,:,26:28]=torch.tensor([1.,0.])
    if frozen: obs[:,:3,20]=1
    nxt=obs.clone(); nxt[:,:,:3]+=.1
    gates=torch.ones(n,4)
    if frozen: gates[:,:3]=0
    values=([obs[:,i].clone() for i in range(4)],[torch.zeros(n,3) for _ in range(4)],
            [torch.linspace(-.2,.3,n) for _ in range(4)],[nxt[:,i].clone() for i in range(4)],
            [torch.full((n,),float(terminal)) for _ in range(4)],torch.ones(n),torch.arange(n),torch.zeros(n,dtype=torch.bool))
    return ExecutionSample(values,gates.clone(),torch.zeros_like(gates) if terminal else gates.clone())

def mutate(source, field, value):
    values=copy.deepcopy(list(source)); values[field]=value
    return ExecutionSample(values,source.enabled.clone(),source.next_enabled.clone())

def state(net): return {k:v.detach().clone() for k,v in net.state_dict().items()}

def delta(before,net): return max(float((before[k]-v).abs().max()) for k,v in net.state_dict().items())
