@echo off
rem the speed of the kernels on the model shapes: test-backend-ops perf of KERNOPS for each arm of KERNARMS, as
rem the int8 coopmat tiles of the 395. no model needed. a result stays until the sources of the build change
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_run.py" kernels
exit /b %ERRORLEVEL%
