@echo off
setlocal
cd /d "%~dp0.."
if errorlevel 1 exit /b 1
set "TASK_CONFIG=%~1"
if not defined TASK_CONFIG set "TASK_CONFIG=configs\chapter3\hgr_decision_experiment.json"
if not "%~4"=="" goto usage
if not "%~2"=="" if not "%~2"=="--execute" goto usage
if not "%~3"=="" if not "%~3"=="--diagnostics-only" goto usage
if not "%~3"=="" if "%~2"=="" goto usage
if not defined PYTHON set "PYTHON=python"
if "%~2"=="--execute" goto execute
"%PYTHON%" -B -m scripts.hgr_decision_experiment --config "%TASK_CONFIG%"
exit /b %ERRORLEVEL%
:execute
if "%~3"=="--diagnostics-only" goto diagnostics
"%PYTHON%" -B -m scripts.hgr_decision_experiment --config "%TASK_CONFIG%" --execute
exit /b %ERRORLEVEL%
:diagnostics
"%PYTHON%" -B -m scripts.hgr_decision_experiment --config "%TASK_CONFIG%" --execute --diagnostics-only
exit /b %ERRORLEVEL%
:usage
echo Usage: %~nx0 [config.json] [--execute] [--diagnostics-only]
echo The default is preflight only. --execute must be the second argument.
exit /b 2
