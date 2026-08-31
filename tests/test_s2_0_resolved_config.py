from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from tests.prrac_evaluation_support import checkpoint_payload


class ResolvedConfigTests(unittest.TestCase):
    def test_cli_paths_are_frozen_after_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); checkpoint=root/"synthetic.pt"; output=root/"result"
            import torch
            torch.save(checkpoint_payload(),checkpoint)
            with mock.patch.object(evaluator,"_evaluate_episode_job",side_effect=lambda job:{"episode":synthetic_row(job),"failure_trace":[],"trace_index":None}), mock.patch.object(evaluator,"ProcessPoolExecutor",ImmediateExecutor):
                evaluator.run_evaluation(checkpoints=[checkpoint],output_dir=output,episodes_override=1,workers_override=1,disable_failure_trace=True)
            import json
            config=json.loads((output/"resolved_evaluation_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["resolved_checkpoint_paths"],[str(checkpoint.resolve())]); self.assertEqual(config["resolved_output_dir"],str(output.resolve()))
            self.assertEqual(config["checkpoints"],config["resolved_checkpoint_paths"]); self.assertEqual(config["output_dir"],config["resolved_output_dir"])


class ImmediateExecutor:
    def __init__(self,*args,**kwargs): pass
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def map(self,function,jobs): return [function(job) for job in jobs]


def synthetic_row(job):
    info=dict(job["checkpoint_info"]); scenario=job["scenario"]
    row={**info,"scenario_id":str(scenario["scenario_id"]),"scenario_seed":int(scenario["scenario_seed"]),"found":False,"contact_episode":False,"hold_episode":False,"success":False,"collision_episode":False,"router_confusion_matrix":[[0,0,0],[0,0,0],[0,0,0]],"searcher_collision_episode_pre_found":False,"searcher_collision_count_pre_found_total":0,"searcher_collision_max_streak_pre_found":0,"searcher_distance_travelled_pre_found":0.0,"map_known_fraction_gain_pre_found":0.0,"search_recovery_entry_count":0,"search_recovery_active_rate":0.0,"executor_collision_count_post_found":0,"post_found_safe_hold_step_count":0,"post_found_route_inactive_step_count":0,"max_steps":1}
    return row


if __name__ == "__main__": unittest.main()
