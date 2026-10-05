@echo off
rem prompt and generation speed at depth: llama-bench, one run per arm of BENCHARMS and depth of BENCHDEPTHS.
rem the arm stable @stable needs 00-build.bat stable once. llama-bench has no expert cache and no MTP, the
rem server numbers of the 3090 come from 04-decode
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_run.py" bench
exit /b %ERRORLEVEL%
