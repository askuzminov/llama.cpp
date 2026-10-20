@echo off
rem where the time goes on the server: a chat reaches each depth of PROFPREFIX (chars) with the prompt cache, then a
rem part of PROFPART chars (pp) and a reply of PROFTG tokens (tg, one decode step); per op from GGML_VK_PERF_LOGGER
rem on vulkan, GGML_CUDA_OP_TIMING=idle on cuda
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_run.py" profile
exit /b %ERRORLEVEL%
