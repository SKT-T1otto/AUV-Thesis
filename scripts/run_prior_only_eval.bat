@echo off
setlocal
rem Keep the paired JSON configuration identical; separate output at launch.
call "%~dp0run_phase1c_prrac_eval.bat" --config "%~dp0..\configs\chapter3\prior_only_eval.json" --output-dir "%~dp0..\outputs\chapter3\phase1c_prrac\prior_only_evaluation_v1" %*
exit /b %ERRORLEVEL%
