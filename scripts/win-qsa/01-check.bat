@echo off
rem correctness of the ops this fork changed: test-backend-ops on the gpu against the cpu, CHECKOPS x
rem CHECKARMS. no model needed. a result stays until the sources of the build change (README.md)
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_run.py" check
exit /b %ERRORLEVEL%
