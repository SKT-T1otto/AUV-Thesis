@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_phase1c_prrac_execution_ablation.ps1" %*
exit /b %ERRORLEVEL%
