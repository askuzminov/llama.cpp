@echo off
rem baseline prefill speed at growing context depth, current graph (no grouping).
rem the gpu shares the system memory, so a step that fills it (02, or any other model run)
rem leaves nothing of the model in the page cache and the next run reads it back from disk
rem while it measures. one prefill is discarded first to pay that once, outside the numbers.
setlocal
call "%~dp0_config.bat"

if not exist "%BIN%\llama-bench.exe" (
    echo not built: %BIN%\llama-bench.exe
    echo run 00-build.bat first
    exit /b 1
)

if not exist "%MODEL%" (
    echo model not found: %MODEL%
    echo set MODEL in _local.bat
    exit /b 1
)

if not defined REPS set "REPS=1"

set "LOG=%LOGS%\03-prefill-%TS%.log"
echo writing %LOG%

echo ### warmup, discarded > "%LOG%"
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p %NPROMPT% -n 0 -b 4096 -ub 512 %LOADMODE% -r 1 %EXTRA% -o md >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ### sweep >> "%LOG%"
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p %NPROMPT% -n 0 -b 4096 -ub %UBATCH% -d %DEPTHS% %LOADMODE% -r %REPS% %EXTRA% --progress -o md >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo exit=%RC% >> "%LOG%"

type "%LOG%"
exit /b %RC%
