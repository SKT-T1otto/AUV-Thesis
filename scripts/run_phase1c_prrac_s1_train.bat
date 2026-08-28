@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_phase1c_prrac_s1_train.ps1" %*
exit /b %ERRORLEVEL%
