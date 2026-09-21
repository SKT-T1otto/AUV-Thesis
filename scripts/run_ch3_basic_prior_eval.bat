@echo off
setlocal
rem OpenMP runtime compatibility (explicit user opt-in only).
if defined KMP_DUPLICATE_LIB_OK (
    echo OPENMP_RUNTIME_ENV >&2
    set KMP_DUPLICATE_LIB_OK >&2
    set KMP_DUPLICATE_LIB_OK | "%SystemRoot%\System32\findstr.exe" /i /x /c:"KMP_DUPLICATE_LIB_OK=FALSE" /c:"KMP_DUPLICATE_LIB_OK=0" /c:"KMP_DUPLICATE_LIB_OK=NO" /c:"KMP_DUPLICATE_LIB_OK=OFF" >nul
    if errorlevel 1 echo WARNING: OpenMP duplicate-runtime bypass explicitly requested; this unsafe workaround may crash or silently produce incorrect results. Numerical reliability is not guaranteed. >&2
) else (
    echo OPENMP_RUNTIME_ENV KMP_DUPLICATE_LIB_OK=^<unset^> >&2
)
rem End OpenMP runtime compatibility.
pushd "%~dp0.." || exit /b 1
python -B -m tools.ch3_baselines.evaluate %*
set "baseline_exit_code=%ERRORLEVEL%"
popd
exit /b %baseline_exit_code%
