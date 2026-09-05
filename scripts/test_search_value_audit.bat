@echo off
setlocal
pushd "%~dp0.." || exit /b 1
if not defined AUV_AUDIT_PYTHON set "AUV_AUDIT_PYTHON=python"
"%AUV_AUDIT_PYTHON%" -m compileall -q chapter3_bser scripts tests
if errorlevel 1 goto failed
"%AUV_AUDIT_PYTHON%" -m unittest discover -s tests -p "test_search_value_audit*.py"
if errorlevel 1 goto failed
"%AUV_AUDIT_PYTHON%" -m unittest tests.test_search_value_head tests.test_search_value_guided_ranking tests.test_prrac_checkpoint_evaluator tests.test_s2_0_evaluation_provenance tests.test_prrac_runtime_factory tests.test_prrac_native_train_eval_runtime_equivalence
if errorlevel 1 goto failed
"%AUV_AUDIT_PYTHON%" -m unittest discover -s tests -p "test_s2a1*.py"
if errorlevel 1 goto failed
popd
exit /b 0
:failed
popd
exit /b 1
