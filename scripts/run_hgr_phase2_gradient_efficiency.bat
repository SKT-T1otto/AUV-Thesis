@echo off
setlocal
cd /d "%~dp0.."
if errorlevel 1 exit /b 1
set "TASK_CONFIG=%~1"
if not defined TASK_CONFIG set "TASK_CONFIG=configs\chapter3\hgr_phase2_gradient_efficiency.json"
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" -m chapter3_bser.experiments.hgr.phase2_gradient_efficiency --config "%TASK_CONFIG%"
if errorlevel 1 exit /b 1
exit /b 0
