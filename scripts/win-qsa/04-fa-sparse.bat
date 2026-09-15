@echo off
rem the same prefill sweep with the sparse flash attention path on and off, one log per arm
rem plus a summary. the selection of cells is the same either way, so this is a pure speed
rem question: does the gather beat the dense kernel on a real per-token mask. the gather also
rem drops the query tile to one row on prefill, so it can lose.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not defined FAVARIANTS set "FAVARIANTS=1 0"
rem порог, с которого префил переходит на тайл в одну строку: KV >= ratio * n_kv_max.
rem 8 это умолчание сборки и его меряет плечо sparse=1, тут только остальные точки
if not defined FARATIOS set "FARATIOS=4 16"
rem плечо gather: пре-пасс объединяет строки маски всего тайла в один список, тайл остаётся
rem многострочным и матмул на coopmat сохраняется. список длиннее, качество то же
if not defined FAGROUP set "FAGROUP=1"

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
for %%r in (%FARATIOS%) do call :run 1 %%r
if "%FAGROUP%"=="1" call :run g

set "GGML_VK_FA_SPARSE_DISABLE="
set "GGML_VK_FA_SPARSE_ROW_RATIO="
set "GGML_VK_FA_SPARSE_GROUP="

echo.
echo summary in %SUM%
type "%SUM%"
exit /b %RC%

rem %1 = 1 gather on, 0 gather off, g объединение строк тайла, %2 = порог тайла в одну строку
:run
set "TAG=s%~1"
if not "%~2"=="" set "TAG=s%~1-r%~2"
set "LOG=%LOGS%\04-fa-sparse-%TS%-%TAG%.log"
set "GGML_VK_FA_SPARSE_DISABLE="
set "GGML_VK_FA_SPARSE_GROUP="
if "%~1"=="0" set "GGML_VK_FA_SPARSE_DISABLE=1"
if "%~1"=="g" set "GGML_VK_FA_SPARSE_GROUP=1"
set "GGML_VK_FA_SPARSE_ROW_RATIO=%~2"
echo === sparse=%~1 ratio=%~2 -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p %NPROMPT% -n 0 -b 4096 -ub %UBATCH% -d %DEPTHS% %LOADMODE% -r %REPS% %EXTRA% --progress -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo. >> "%SUM%"
echo ### sparse=%~1 ratio=%~2 exit=%EC% >> "%SUM%"
type "%LOG%" >> "%SUM%"
goto :eof
