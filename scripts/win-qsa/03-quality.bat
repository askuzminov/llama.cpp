@echo off
rem how far each arm of QARMS moves the output distribution from the reference QREF: llama-perplexity, PPL
rem ratio and KLD over QCTX x QCHUNKS tokens of PPLFILE. the reference logits stay in cache\ref
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_run.py" quality
exit /b %ERRORLEVEL%
