@echo off
rem a timeline of one conversation on the server with Nsight Systems (CUDA only): the kernels, the uploads of the
rem experts and the idle time of the GPU in each prompt ubatch and in the decode steps (TRACECHAT, TRACEGEN, TRACEARMS)
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_run.py" trace
exit /b %ERRORLEVEL%
