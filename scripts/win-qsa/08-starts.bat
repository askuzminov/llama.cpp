@echo off
rem how often the server starts: STARTS starts of each arm of STARTARMS with the arguments of 04, one try each,
rem the stalled loads counted (the 395: the upload into the MTP head buffer past the 96 GiB carve-out)
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_run.py" starts
exit /b %ERRORLEVEL%
