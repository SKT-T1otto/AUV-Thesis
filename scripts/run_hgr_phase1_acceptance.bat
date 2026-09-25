@echo off
setlocal
cd /d "%~dp0.."
if errorlevel 1 exit /b 1
set "TASK_OUTPUT=%~1"
if not defined TASK_OUTPUT set "TASK_OUTPUT=outputs\chapter3\hgr_phase1\collision_terminal\manual_acceptance_01"
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" -m chapter3_bser.experiments.hgr.phase1_acceptance --config configs/chapter3/hgr_phase1_zero.json --output "%TASK_OUTPUT%"
if errorlevel 1 exit /b 1
exit /b 0
