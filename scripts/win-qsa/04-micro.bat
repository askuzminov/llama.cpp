@echo off
rem the QSA indexer alone on the device: the lightning indexer and the head matmuls at the token counts of decode,
rem MTP checks and prompts and depths to the end of the context, time per graph (test-micro-fork, README.md)
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_qa.py" micro
exit /b %ERRORLEVEL%
