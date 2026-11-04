@echo off
rem the ops of the model on the device against the cpu, the unit tests of the fork, the logits at a few depths of one
rem context (32768 tokens) through the prompt path and the decode path against a slower, more exact run of the same
rem build, a restore of a partial checkpoint against the run it continues, the speculation of the server. about 20
rem minutes. a result stays until the sources of the build change (README.md)
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_qa.py" check
exit /b %ERRORLEVEL%
