@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_ch3_learning.ps1" %*
exit /b %errorlevel%
