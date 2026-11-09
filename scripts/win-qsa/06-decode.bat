@echo off
rem decode speed by kind of text (prose, new code, a file edit, JSON) on the server with the MTP head, variants of
rem QA_DARMS side by side. about 5 minutes a variant on the 395 (README.md)
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_qa.py" decode
exit /b %ERRORLEVEL%
