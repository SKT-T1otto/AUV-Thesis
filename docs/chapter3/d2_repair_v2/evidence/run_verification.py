"""Record full selected unittest results and command/environment, without skips."""
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import unittest
import torch

TARGETED='''test_maddpg_sample_contract test_ch3_replay_restore_contract test_execution_replay_gate_contract test_maddpg_terminal_target_numerics test_maddpg_actor_update_mask test_waypoint_reward_timing test_d2_repair_v2_provenance test_d2_repair_v2_ablation test_d2_repair_v2_runtime_contract'''.split()
REGRESSION='''test_core_checkpoint_roundtrip test_core_has_no_legacy_imports test_core_import_graph test_core_scenario_generation test_core_source_provenance test_core_standalone_env test_core_training_smoke test_core_without_legacy_directory test_mission_core_contract test_observation_28d_contract test_collision_terminal_protocol test_phase1a_core_freeze test_phase1a1_original_core_freeze test_phase1c_guidance test_phase1b2_path_tracking test_phase1c_v2_training_smoke test_phase1c_v2_reward_protocol test_phase1c_v2_execution_reward test_phase1c_v2_replay test_prrac_replay test_prrac_native_train_eval_runtime_equivalence test_prrac_b1_native_training_runtime test_prrac_training_smoke test_prrac_isolation test_team_reward test_ch3_baseline_training test_ch3_baseline_validation test_d2_implementation_repairs test_repository_metadata test_d2_target_prediction_cache'''.split()

class Result(unittest.TextTestResult):
    def __init__(self,*a,**kw): super().__init__(*a,**kw); self.successes=[]
    def addSuccess(self,test): super().addSuccess(test); self.successes.append(test.id())

def main():
    torch.set_num_threads(1); started=time.perf_counter(); scope=sys.argv[1]
    modules=TARGETED if scope=='targeted' else REGRESSION
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.'+m for m in modules])
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    record=dict(scope=scope,command=' '.join(sys.argv),modules=modules,tests_run=result.testsRun,
        successes=result.successes,failures=[dict(test=t.id(),traceback=tb) for t,tb in result.failures],
        errors=[dict(test=t.id(),traceback=tb) for t,tb in result.errors],skipped=[dict(test=t.id(),reason=r) for t,r in result.skipped],
        exit_code=0 if result.wasSuccessful() else 1,wall_seconds=time.perf_counter()-started,
        python=sys.version,executable=sys.executable,platform=platform.platform(),torch=torch.__version__,
        threads=torch.get_num_threads(),environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},is_ci=False)
    Path(f'tmp/d2_repair_v2_20261018/{scope}.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps(dict(scope=scope,run=result.testsRun,success=len(result.successes),errors=len(result.errors),failures=len(result.failures))))
    raise SystemExit(record['exit_code'])

if __name__=='__main__': main()
