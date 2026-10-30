@echo off
rem the ops of the model on the device against the cpu, the unit tests of the fork, and the logits at a few depths of
rem one context (32768 tokens) through the prompt path and the decode path against a slower, more exact run of the
rem same build. about 10 minutes. a result stays until the sources of the build change (README.md)
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_qa.py" check
exit /b %ERRORLEVEL%
