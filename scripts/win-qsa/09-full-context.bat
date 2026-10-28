@echo off
rem Full native context: quality, MTP, parallelism, prefill, cache and profiling.
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_run.py" full
exit /b %ERRORLEVEL%
