@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_collision_terminal.ps1" %*
exit /b %errorlevel%
