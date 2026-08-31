@echo off
rem Fixed evaluator variants: S2A_C0_BASELINE S2A_C1_ROUTE_REFRESH S2A_C2_EGRESS_ROUTE
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_phase1c_prrac_s2a_collision_ablation.ps1" %*
exit /b %ERRORLEVEL%
