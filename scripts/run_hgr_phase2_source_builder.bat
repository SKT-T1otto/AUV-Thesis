@echo off
setlocal
cd /d "%~dp0.."
if errorlevel 1 exit /b 1
set "TASK_CONFIG=%~1"
if not defined TASK_CONFIG set "TASK_CONFIG=configs\chapter3\hgr_phase2_source_builder.json"
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" -m chapter3_bser.experiments.hgr.build_phase2_source --config "%TASK_CONFIG%"
set "TASK_EXIT=%ERRORLEVEL%"
exit /b %TASK_EXIT%
