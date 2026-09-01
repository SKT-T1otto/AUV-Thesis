@echo off
setlocal EnableExtensions
set "CHECKPOINT="
set "OUTPUT_DIR=outputs/chapter3/phase1c_prrac/s2a1_local_connector_ablation"
set "WORKERS=4"
set "EPISODES=10"
set "FORMAL=0"
set "SCENARIO_ID_FILE="
:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--checkpoint" set "CHECKPOINT=%~2"& shift& shift& goto parse
if /I "%~1"=="--output-dir" set "OUTPUT_DIR=%~2"& shift& shift& goto parse
if /I "%~1"=="--workers" set "WORKERS=%~2"& shift& shift& goto parse
if /I "%~1"=="--episodes" set "EPISODES=%~2"& shift& shift& goto parse
if /I "%~1"=="--scenario-id-file" set "SCENARIO_ID_FILE=%~2"& shift& shift& goto parse
if /I "%~1"=="--formal" set "FORMAL=1"& shift& goto parse
echo ERROR: unknown argument %~1 1>&2
exit /b 2
:parsed
if not defined CHECKPOINT echo ERROR: --checkpoint is required 1>&2& exit /b 2
if "%FORMAL%"=="1" if defined SCENARIO_ID_FILE echo ERROR: --formal cannot be combined with --scenario-id-file 1>&2& exit /b 2
where conda >nul 2>nul || (echo ERROR: conda was not found on PATH 1>&2& exit /b 1)
set "MPLBACKEND=Agg"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
set "ARGS=--config configs/chapter3/bser_phase1c_prrac_s2a1_local_connector_ablation.json --checkpoint "%CHECKPOINT%" --output-dir "%OUTPUT_DIR%" --workers %WORKERS% --modes full_prrac --execution-variants B1_ATOMIC_LAST_VALID --search-recovery-variants S2A1_C0_BASELINE S2A1_C1_FORCED_REFRESH S2A1_C2_LOCAL_CONNECTOR"
if "%FORMAL%"=="1" (set "ARGS=%ARGS% --formal") else if defined SCENARIO_ID_FILE (set "ARGS=%ARGS% --scenario-id-file "%SCENARIO_ID_FILE%"") else (set "ARGS=%ARGS% --episodes %EPISODES%")
call conda run --no-capture-output -n AUV python -B -m chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints %ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
