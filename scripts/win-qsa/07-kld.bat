@echo off
rem the logits of this build against those of upstream (00-build.bat upstream first) on the same model and text:
rem KLD, top-1 agreement and the perplexity ratio for each arm of QA_KARMS, next to the last result of the arm from
rem another build (after an upstream merge: did the merge move the fork away from upstream). README.md
setlocal
call "%~dp0_config.bat"
if not defined PY (
    echo Python 3 is required: py -3 or python on PATH
    exit /b 1
)
%PY% "%~dp0_qa.py" kld
exit /b %ERRORLEVEL%
