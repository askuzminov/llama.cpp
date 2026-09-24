@echo off
rem per-op timings from the vulkan backend at a deep context, one run per arm of FAVARIANTS.
rem produces a lot of output, keep the run short. cuda ignores GGML_VK_PERF_LOGGER.
rem the point of the sweep: the gather path walks the list of cells a query tile can see, the
rem dense path walks all of KV. the FLASH_ATTN_EXT line says which one wins on the real
rem per-token mask, and the lines around it say what the pre-pass costs. arms g and r are the
rem two tile shapes of the gather, see 04-fa-sparse.bat.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem весь шаг это GGML_VK_PERF_LOGGER, на другом бэкенде он даст только загрузку модели
if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 07-perf-logger is vulkan only
    exit /b 0
)

if not defined FAVARIANTS set "FAVARIANTS=1 0 g r"
if not defined SETTLE set "SETTLE=30"
rem depth of the measured graph. the sparsity of the indexer grows with it: the budget is a
rem fixed number of blocks, so the deeper the context the larger the share of skippable tiles
if not defined PERFDEPTH set "PERFDEPTH=64000"

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

set "SUM=%LOGS%\07-perf-%TS%-summary.txt"
echo writing %SUM%

set "GGML_VK_PERF_LOGGER=1"
set "RC=0"
set "FIRST=1"

for %%v in (%FAVARIANTS%) do call :run %%v

set "GGML_VK_PERF_LOGGER="
set "GGML_VK_FA_SPARSE_DISABLE="
set "GGML_VK_FA_SPARSE_GROUP="

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = плечо, как в 04: 0 плотное ядро, 1 умолчание, g объединение, r тайл в одну строку
:run
if "%FIRST%"=="1" (set "FIRST=0") else (
    echo === waiting %SETTLE%s for the gpu to be released
    powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
)
set "LOG=%LOGS%\07-perf-%TS%-s%~1.log"
set "GGML_VK_FA_SPARSE_DISABLE="
set "GGML_VK_FA_SPARSE_GROUP="
if "%~1"=="0" set "GGML_VK_FA_SPARSE_DISABLE=1"
if "%~1"=="g" set "GGML_VK_FA_SPARSE_GROUP=1"
if "%~1"=="r" set "GGML_VK_FA_SPARSE_GROUP=0"
echo === sparse=%~1
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 512 -n 0 -b 4096 -ub 2048 -d %PERFDEPTH% %LOADMODE% -r 1 --no-warmup %EXTRA% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"

echo. >> "%SUM%"
echo ### sparse=%~1 exit=%EC% >> "%SUM%"
findstr /C:"| qwen4exp" /C:"failed to load model" /C:"allocation of size" "%LOG%" >> "%SUM%"

rem only the last graph is measured at the requested depth, the ones before it build the context
powershell -NoProfile -Command "$t = Get-Content -LiteralPath '%LOG%'; $m = $t | Select-String -SimpleMatch 'Vulkan Timings:' | Select-Object -Last 1; if ($m) { $t[($m.LineNumber - 1)..($t.Count - 1)] }" >> "%SUM%"
goto :eof
