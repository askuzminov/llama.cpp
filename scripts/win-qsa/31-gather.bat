@echo off
rem GET_ROWS with short rows (02.10). the sparse attention of qwen4exp gathers the KQ mask value of each selected
rem cell (rows of one element) and the cells of each selected 4-cell block (rows of four). the old dispatch put
rem one row on a 512-wide workgroup, so 511 of 512 invocations had nothing to do: on the Mac 5500M these two
rem gathers took 144 and 36 ms at a 4096-token ubatch, and GET_ROWS was 290 ms of the 5125 ms ubatch in the 30
rem profile. rows shorter than 512 now take a flat dispatch that spreads the elements of all rows over the
rem workgroup (Mac: 0.84 and 0.60 ms). GGML_VK_DISABLE_GET_ROWS_FLAT=1 gives the old dispatch for the comparison.
rem the same change fixes i32 GET_ROWS with rows wider than 512 (the pipeline counted 1024 per workgroup of 512).
rem   1 GET_ROWS against the CPU; a FAIL stops before the model is loaded
rem   2 perf of the two gathers, flat and old
rem   3 with the model: llama-bench pp4096 (ub 2048 and 4096) and tg64, flat and old
rem   4 per-op profile of one ubatch of 4096 (GGML_VK_PERF_LOGGER), flat and old. the flat run also asks for the
rem     register statistics of the matmul pipelines (GGML_VK_PIPELINE_STATS=matmul); the driver gives them only
rem     if it has VK_KHR_pipeline_executable_properties, the summary says if it has none
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 31-gather is vulkan only
    exit /b 0
)

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

set "SUM=%LOGS%\31-gather-%TS%-summary.txt"
set "PERFP=n=1,m=4096|n=4,m=1024"
set "RC=0"
set "TESTRC=0"
set "GGML_VK_DISABLE_GET_ROWS_FLAT="
set "GGML_VK_PERF_LOGGER="
set "GGML_VK_PIPELINE_STATS="
echo writing %SUM%

call :test
if not "%TESTRC%"=="0" goto :failed
rem 0 = flat (the default), 1 = GGML_VK_DISABLE_GET_ROWS_FLAT
for %%f in (0 1) do call :perf %%f

rem paths stay out of parenthesized blocks, a ")" in one would end the block
if not exist "%MODEL%" goto :nomodel
if not exist "%BIN%\llama-bench.exe" goto :nomodel
for %%f in (0 1) do call :bench %%f
for %%f in (0 1) do call :prof %%f
goto :end

:failed
echo ### a test failed, the model runs are skipped >> "%SUM%"
goto :end

:nomodel
echo ### model or llama-bench not found, the model runs are skipped >> "%SUM%"

:end
set "GGML_VK_DISABLE_GET_ROWS_FLAT="
set "GGML_VK_PERF_LOGGER="
set "GGML_VK_PIPELINE_STATS="

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

:test
set "LOG=%LOGS%\31-gather-%TS%-test-GET_ROWS.log"
echo === test GET_ROWS -^> %LOG%
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o GET_ROWS > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
findstr /C:"tests passed" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" set "RC=%EC%"
if not "%EC%"=="0" set "TESTRC=%EC%"
echo ### test GET_ROWS, exit=%EC% >> "%SUM%"
findstr /C:"tests passed" /C:"FAIL" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem the filter holds |, so it stays in quotes and out of parenthesized blocks
:perf
set "LOG=%LOGS%\31-gather-%TS%-perf-%~1.log"
set "GGML_VK_DISABLE_GET_ROWS_FLAT="
if "%~1"=="1" set "GGML_VK_DISABLE_GET_ROWS_FLAT=1"
echo === perf GET_ROWS, flat off %~1 -^> %LOG%
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o GET_ROWS -p "%PERFP%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_DISABLE_GET_ROWS_FLAT="
if not "%EC%"=="0" set "RC=%EC%"
echo ### perf GET_ROWS, flat off %~1, exit=%EC% >> "%SUM%"
findstr /C:"us/run" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem the driver frees the memory of the previous process lazily, and a model load right after it
rem can fail to allocate even after SETTLE (README, traps): one more try after another wait
:bench
set "TRY=0"
:bench_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\31-gather-%TS%-bench-%~1.log"
set "GGML_VK_DISABLE_GET_ROWS_FLAT="
if "%~1"=="1" set "GGML_VK_DISABLE_GET_ROWS_FLAT=1"
echo === llama-bench pp4096 tg64, flat off %~1, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 64 -b 4096 -ub 2048,4096 -d 0 %LOADMODE% -r 2 %EXTRA% -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_DISABLE_GET_ROWS_FLAT="
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :bench_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### llama-bench pp4096 tg64, flat off %~1, try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:" pp4096 " /C:" tg64 " /C:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem the whole per-op table goes to the summary; the register statistics stay in the log, the summary takes
rem the lines with registers, scratch (spills) and shared memory
:prof
set "TRY=0"
:prof_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\31-gather-%TS%-prof-%~1.log"
set "GGML_VK_DISABLE_GET_ROWS_FLAT="
if "%~1"=="1" set "GGML_VK_DISABLE_GET_ROWS_FLAT=1"
if "%~1"=="0" set "GGML_VK_PIPELINE_STATS=matmul"
set "GGML_VK_PERF_LOGGER=1"
echo === per-op profile pp4096 ub4096, flat off %~1, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 0 -b 4096 -ub 4096 -d 0 %LOADMODE% -r 1 --no-warmup %EXTRA% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_DISABLE_GET_ROWS_FLAT="
set "GGML_VK_PERF_LOGGER="
set "GGML_VK_PIPELINE_STATS="
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :prof_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### profile pp4096 ub4096, flat off %~1, try %TRY%, exit=%EC% >> "%SUM%"
findstr /R /C:" x .* us = " /C:"Total time" /C:" pp4096 " /C:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
if not "%~1"=="0" goto :eof
findstr /C:"pipeline stats for" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" goto :nostats
echo ### matmul pipeline statistics (all of them in %LOG%) >> "%SUM%"
findstr /I /C:"pipeline stats for" /C:"gpr" /C:"scratch" /C:"spill" /C:"lds" /C:"shared" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

:nostats
echo ### no pipeline statistics: the driver has no VK_KHR_pipeline_executable_properties >> "%SUM%"
echo. >> "%SUM%"
goto :eof
