@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_phase1c_v2_1_train.ps1" %*
exit /b %ERRORLEVEL%
