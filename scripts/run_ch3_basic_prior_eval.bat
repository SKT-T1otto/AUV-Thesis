@echo off
setlocal
pushd "%~dp0.." || exit /b 1
python -B -m tools.ch3_baselines.evaluate %*
set "baseline_exit_code=%ERRORLEVEL%"
popd
exit /b %baseline_exit_code%
