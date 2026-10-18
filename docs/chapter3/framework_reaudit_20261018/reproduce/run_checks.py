"""Record current relevant module results, with every failure retained."""
import json
import os
import sys
import time
import unittest
from pathlib import Path
import torch

MODULES = '''test_core_checkpoint_roundtrip test_core_has_no_legacy_imports test_core_import_graph test_core_scenario_generation test_core_source_provenance test_core_standalone_env test_core_training_smoke test_core_without_legacy_directory test_mission_core_contract test_observation_28d_contract test_collision_terminal_protocol test_phase1a_core_freeze test_phase1a1_original_core_freeze test_phase1c_guidance test_phase1b2_path_tracking test_phase1c_v2_training_smoke test_phase1c_v2_reward_protocol test_phase1c_v2_execution_reward test_prrac_replay test_prrac_native_train_eval_runtime_equivalence test_prrac_b1_native_training_runtime test_prrac_training_smoke test_prrac_isolation test_team_reward test_ch3_baseline_training test_ch3_baseline_validation test_d2_implementation_repairs test_repository_metadata test_d2_target_prediction_cache'''.split()

class Result(unittest.TextTestResult):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.passed=[]
    def addSuccess(self,test):
        super().addSuccess(test)
        self.passed.append(test.id())

def main():
    Path('tmp/framework_reaudit_20261018').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    started=time.perf_counter()
    # Use the actual existing cache test name, avoiding a synthetic missing-test error.
    MODULES[-1]='test_d2_target_prediction_cache' if Path('tests/test_d2_target_prediction_cache.py').exists() else 'test_d2_target_prediction'
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.'+m for m in MODULES])
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    payload=dict(scope='complete selected core/environment/training modules, not all repository tests',modules=MODULES,
        tests_run=result.testsRun,successes=result.passed,failures=[dict(test=t.id(),traceback=tb) for t,tb in result.failures],
        errors=[dict(test=t.id(),traceback=tb) for t,tb in result.errors],skipped=[dict(test=t.id(),reason=r) for t,r in result.skipped],
        successful=result.wasSuccessful(),wall_seconds=time.perf_counter()-started,
        environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    Path(sys.argv[1] if len(sys.argv)>1 else 'tmp/framework_reaudit_20261018/regression.json').write_bytes((json.dumps(payload,indent=2)+'\n').encode())
    print(json.dumps(dict(run=result.testsRun,successes=len(result.passed),failures=len(result.failures),errors=len(result.errors))))
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__=='__main__': main()
