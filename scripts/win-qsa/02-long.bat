@echo off
rem the whole native context of the model: the logits at depths to the end, one conversation to the end with facts to
rem recall on the way and a long answer at the end, four conversations at once. about 40 minutes on the 395. stop
rem the daily server (serve.bat) first, it holds the gpu memory (README.md)
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_qa.py" long
exit /b %ERRORLEVEL%
