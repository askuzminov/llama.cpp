@echo off
rem where the time of one ubatch goes at PROFDEPTH: per op from GGML_VK_PERF_LOGGER on vulkan, per split from
rem GGML_SCHED_PROF on cuda. PROFKINDS pp is the prompt ubatch, tg one decode step
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_run.py" profile
exit /b %ERRORLEVEL%
