@echo off
rem perplexity of the current build, the reference number for the model. it only means
rem something when the context is well above the qsa budget of about 2051 cells: below it
rem the indexer is bypassed and the run says nothing about the selection.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not defined QCTX    set "QCTX=32768"
if not defined QCHUNKS set "QCHUNKS=4"

if not exist "%BIN%\llama-perplexity.exe" (
    echo not built: %BIN%\llama-perplexity.exe
    echo run 00-build.bat first
    exit /b 1
)

if not exist "%MODEL%" (
    echo model not found: %MODEL%
    echo set MODEL in _local.bat
    exit /b 1
)

if not exist "%PPLFILE%" (
    echo missing %PPLFILE%, run get-wikitext.bat first
    exit /b 1
)

set "SUM=%LOGS%\05-ppl-%TS%-summary.txt"
set "RC=0"

call :run

echo.
type "%SUM%"
exit /b %RC%

:run
set "LOG=%LOGS%\05-ppl-%TS%.log"
echo === perplexity, c=%QCTX% chunks=%QCHUNKS% -^> %LOG%
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c %QCTX% --chunks %QCHUNKS% -fa on %LOADMODE% %EXTRA% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo. >> "%SUM%"
echo ### c=%QCTX% chunks=%QCHUNKS% exit=%EC% >> "%SUM%"
findstr /c:"Final estimate" "%LOG%" >> "%SUM%"
goto :eof
