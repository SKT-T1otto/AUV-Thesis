@echo off
setlocal
pushd "%~dp0.." || exit /b 1
python -B -m tools.ch3_baselines.run_training %* --baseline B3_direct_boundary
set "baseline_exit_code=%ERRORLEVEL%"
popd
exit /b %baseline_exit_code%
