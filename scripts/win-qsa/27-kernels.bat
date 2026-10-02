@echo off
rem kernel work on the prefill of strix halo, step 1: where the time of one ubatch of 4096 goes now (int8
rem coopmat by default since 02.10), and the first kernel experiment, the large int8 coopmat tile
rem (GGML_VK_INT_LARGE_TILE=1: 512 threads, 128x128, 16 waves of 32x32, 4 accumulators each). the AMD
rem driver has it off because the large f16 tile spills there (28.09), but the int8 one holds far fewer
rem accumulators per wave. the dense matmuls keep the medium tile all the same: 128x128 needs about 36 KB
rem of shared memory, the driver gives 32 KB. matmul_id fits (K step 2, about 24 KB).
rem   1 MUL_MAT, MUL_MAT_ID and MUL_MAT_ID_FUSION with the large tile against the CPU; a FAIL stops there
rem   2 perf of the qwen4exp shapes at 4096 tokens, default and large tile
rem   3 with the model: llama-bench pp4096 (ub 2048 and 4096), default and large tile
rem   4 per-op profile of one ubatch of 4096 (GGML_VK_PERF_LOGGER), default and large tile
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 27-kernels is vulkan only
    exit /b 0
)

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

set "SUM=%LOGS%\27-kernels-%TS%-summary.txt"
rem the quants with an int8 pipeline on RDNA3
set "TQP=type_a=(q4_0|q4_1|q5_0|q5_1|q8_0|iq4_nl|iq4_xs|mxfp4|q3_K|q4_K|q5_K|q6_K),"
rem the perf cases of 25-tile-sweep.bat: the q8_0 projections and the experts at 4096 tokens
set "MMP=type_a=q8_0,type_b=f32,m=(320|640|2560|6144|10240|12288),n=4096,k=(320|2560|6144|10240),bs=\[1,1\]"
set "IDP=n_mats=512,n_used=10,b=0,m=(640|2560),n=4096"
set "RC=0"
set "TESTRC=0"
set "GGML_VK_INT_LARGE_TILE="
set "GGML_VK_PERF_LOGGER="
echo writing %SUM%

call :test MUL_MAT
call :test MUL_MAT_ID
call :test MUL_MAT_ID_FUSION
if not "%TESTRC%"=="0" goto :failed

rem 0 = the default, 1 = the large int8 tile
for %%t in (0 1) do call :perf %%t

rem paths stay out of parenthesized blocks, a ")" in one would end the block
if not exist "%MODEL%" goto :nomodel
if not exist "%BIN%\llama-bench.exe" goto :nobench
for %%t in (0 1) do call :bench %%t
for %%t in (0 1) do call :prof %%t
goto :end

:failed
echo ### a test failed with the large int8 tile, perf and model runs skipped >> "%SUM%"
goto :end

:nomodel
echo ### model not found: %MODEL%, the model runs are skipped >> "%SUM%"
goto :end

:nobench
echo ### not built: %BIN%\llama-bench.exe, the model runs are skipped >> "%SUM%"

:end
set "GGML_VK_INT_LARGE_TILE="
set "GGML_VK_PERF_LOGGER="

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = the op to test, with the large int8 tile
:test
set "LOG=%LOGS%\27-kernels-%TS%-test-%~1.log"
set "GGML_VK_INT_LARGE_TILE=1"
echo === test %~1, large int8 tile -^> %LOG%
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o %~1 -p "%TQP%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_INT_LARGE_TILE="
findstr /C:"tests passed" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" set "RC=%EC%"
if not "%EC%"=="0" set "TESTRC=%EC%"
echo ### test %~1, large int8 tile, exit=%EC% >> "%SUM%"
findstr /C:"int8 coopmat" /C:"tests passed" /C:"FAIL" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem %1 = 0 default, 1 large int8 tile
:perf
set "LOG=%LOGS%\27-kernels-%TS%-perf-%~1.log"
set "GGML_VK_INT_LARGE_TILE=%~1"
echo === perf, large int8 tile %~1 -^> %LOG%
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT -p "%MMP%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT_ID -p "%IDP%" >> "%LOG%" 2>&1
set "EC2=%ERRORLEVEL%"
set "GGML_VK_INT_LARGE_TILE="
if not "%EC%"=="0" set "RC=%EC%"
if not "%EC2%"=="0" set "RC=%EC2%"
echo ### perf, large int8 tile %~1, exit=%EC% %EC2% >> "%SUM%"
findstr /C:"MUL_MAT(" /C:"MUL_MAT_ID(" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem the driver frees the memory of the previous process lazily, and a model load right after it
rem can fail to allocate even after SETTLE (README, traps). every model run below gets one more
rem try after another wait when its log says "failed to allocate" or "Device memory allocation"
:bench
set "TRY=0"
:bench_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\27-kernels-%TS%-bench-%~1.log"
set "GGML_VK_INT_LARGE_TILE=%~1"
echo === llama-bench pp4096, large int8 tile %~1, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 0 -b 4096 -ub 2048,4096 -d 0 %LOADMODE% -r 2 %EXTRA% -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_INT_LARGE_TILE="
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :bench_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### llama-bench pp4096, large int8 tile %~1, try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:" pp4096 " /C:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem the whole per-op table goes to the summary: it is the map for the next kernel steps
:prof
set "TRY=0"
:prof_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\27-kernels-%TS%-prof-%~1.log"
set "GGML_VK_INT_LARGE_TILE=%~1"
set "GGML_VK_PERF_LOGGER=1"
echo === per-op profile pp4096 ub4096, large int8 tile %~1, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 0 -b 4096 -ub 4096 -d 0 %LOADMODE% -r 1 --no-warmup %EXTRA% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_INT_LARGE_TILE="
set "GGML_VK_PERF_LOGGER="
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :prof_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### profile pp4096 ub4096, large int8 tile %~1, try %TRY%, exit=%EC% >> "%SUM%"
findstr /R /C:" x .* us = " /C:"Total time" /C:" pp4096 " /C:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
