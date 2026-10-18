"""Bounded adjacent checkpoint contracts and plan-only historical Suite checks."""
import sys
from tmp.d2_repair_v2_20261018 import run_verification as runner

runner.REGRESSION=[
    'test_prrac_checkpoint',
    'test_prrac_runtime_checkpoint_compatibility',
    'test_prrac_checkpoint_evaluator',
    'test_phase1c_checkpoint_evaluator',
    'test_d2_suite_pipeline.PipelineTests.test_default_nine_training_ten_evaluation_preview_writes_nothing',
    'test_d2_suite_linux.LinuxLaunchTests.test_native_preflight_and_previews_preserve_plan_and_start_nothing',
    'test_d2_suite_linux_evaluation.EvaluationEntryTests.test_previews_select_reference_or_learning_jobs_without_writing',
]
sys.argv=[__file__,'supplemental']
runner.main()
