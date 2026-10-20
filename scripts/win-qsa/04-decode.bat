@echo off
rem generation through llama-server, one start per arm of DECARMS: MTP, the expert cache, the cpu overlap.
rem DECPROMPTS x DECSEEDS requests of DECNGEN tokens, DECPREFIX chars of PPLFILE in front of each prompt for
rem a long one. stops any run when another llama-server.exe is up: it holds gpu memory
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_run.py" decode
exit /b %ERRORLEVEL%
