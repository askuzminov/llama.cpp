@echo off
rem the same prefill sweep with the sparse flash attention path on and off, one log per value
rem of GGML_VK_FA_SPARSE plus a summary. the selection of cells is the same either way, so this
rem is a pure speed question: does the gather beat the dense kernel on a real per-token mask.
rem the gather also gives up the aligned kernel, so it can lose.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not defined FAVARIANTS set "FAVARIANTS=1 0"

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

set "SUM=%LOGS%\04-fa-sparse-%TS%-summary.txt"
set "RC=0"

for %%v in (%FAVARIANTS%) do call :run %%v

set "GGML_VK_FA_SPARSE="

echo.
echo summary in %SUM%
type "%SUM%"
exit /b %RC%

rem %1 = value of GGML_VK_FA_SPARSE
:run
set "LOG=%LOGS%\04-fa-sparse-%TS%-s%~1.log"
set "GGML_VK_FA_SPARSE=%~1"
echo === GGML_VK_FA_SPARSE=%~1 -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p %NPROMPT% -n 0 -b 4096 -ub %UBATCH% -d %DEPTHS% %LOADMODE% -r %REPS% %EXTRA% --progress -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo. >> "%SUM%"
echo ### GGML_VK_FA_SPARSE=%~1 exit=%EC% >> "%SUM%"
type "%LOG%" >> "%SUM%"
goto :eof
