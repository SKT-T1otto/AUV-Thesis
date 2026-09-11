@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_prrac_controller_pair.ps1" -Episodes 10 %*
exit /b %ERRORLEVEL%
