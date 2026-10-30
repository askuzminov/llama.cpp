@echo off
rem where the time goes at depths to the end of the context: gpu time per kind of op for a prompt ubatch and a decode
rem step, the host steps, the MTP head. about 15 minutes on the 395 (README.md)
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_qa.py" profile
exit /b %ERRORLEVEL%
