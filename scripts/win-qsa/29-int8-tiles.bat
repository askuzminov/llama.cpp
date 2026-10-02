@echo off
rem kernel work on the prefill of strix halo, step 2: 64 KB of shared memory is the default on the RDNA3
rem iGPU of the AMD driver since 02.10 (outside the Vulkan spec, GGML_VK_SHMEM_LIMIT=32768 keeps the 32 KB
rem the driver reports), and the large int8 coopmat tile can grow with it.
rem   1 the 64 KB default against 32 KB on the model: llama-bench pp4096 and tg64 at depth 0 and 32768
rem     (flash attention picks its tiles by the shared memory, decode included)
rem   2 sweep of the large int8 tile (GGML_VK_MMQ_INT_TILE "BM,BN,WM,WN", INT8TILES in _config.bat) with
rem     the large int8 tiles on (GGML_VK_INT_LARGE_TILE=1): per tile MUL_MAT, MUL_MAT_ID and
rem     MUL_MAT_ID_FUSION against the CPU for the quants of qwen4exp and perf of its shapes at 4096 tokens.
rem     the medium tile (no large tile) runs first as the reference. a tile only counts when its tests pass
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 29-int8-tiles is vulkan only
    exit /b 0
)

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

set "SUM=%LOGS%\29-int8-tiles-%TS%-summary.txt"
rem the quants of qwen4exp
set "TQP=type_a=(q8_0|q4_K|q5_K|q5_1),"
rem the perf cases of 25-tile-sweep.bat: the q8_0 projections and the experts at 4096 tokens
set "MMP=type_a=q8_0,type_b=f32,m=(320|640|2560|6144|10240|12288),n=4096,k=(320|2560|6144|10240),bs=\[1,1\]"
set "IDP=n_mats=512,n_used=10,b=0,m=(640|2560),n=4096"
set "RC=0"
set "N=0"
call :clear
echo writing %SUM%

rem paths stay out of parenthesized blocks, a ")" in one would end the block
if not exist "%MODEL%" goto :nomodel
if not exist "%BIN%\llama-bench.exe" goto :nomodel
call :bench 0
call :bench 32768
goto :sweep

:nomodel
echo ### model or llama-bench not found, the 64 KB check on the model is skipped >> "%SUM%"

:sweep
call :tile medium
for %%t in (%INT8TILES%) do call :tile %%t

call :clear
echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

:clear
set "GGML_VK_SHMEM_LIMIT="
set "GGML_VK_INT_LARGE_TILE="
set "GGML_VK_MMQ_INT_TILE="
goto :eof

rem %1 = the shared memory limit, 0 = the default (64 KB here)
:bench
set "TRY=0"
:bench_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\29-int8-tiles-%TS%-bench-%~1.log"
call :clear
if not "%~1"=="0" set "GGML_VK_SHMEM_LIMIT=%~1"
echo === llama-bench pp4096 tg64 at depth 0 and 32768, shared memory limit %~1, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 64 -b 4096 -ub 2048 -d 0,32768 %LOADMODE% -r 1 %EXTRA% -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :clear
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :bench_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### llama-bench, shared memory limit %~1 (0 = default), try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:" pp4096" /C:" tg64" /C:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem %1 = "BM,BN,WM,WN" in quotes, or medium = no large tile
:tile
set /a N+=1
set "T=%~1"
call :clear
if /i not "%T%"=="medium" set "GGML_VK_INT_LARGE_TILE=1"
if /i not "%T%"=="medium" set "GGML_VK_MMQ_INT_TILE=%T%"
set "TLOG=%LOGS%\29-int8-tiles-%TS%-%N%-test.log"
set "PLOG=%LOGS%\29-int8-tiles-%TS%-%N%-perf.log"
echo === tile %N%: %T%
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o MUL_MAT -p "%TQP%" > "%TLOG%" 2>&1
set "E1=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o MUL_MAT_ID -p "%TQP%" >> "%TLOG%" 2>&1
set "E2=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o MUL_MAT_ID_FUSION -p "%TQP%" >> "%TLOG%" 2>&1
set "E3=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT -p "%MMP%" > "%PLOG%" 2>&1
set "E4=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT_ID -p "%IDP%" >> "%PLOG%" 2>&1
set "E5=%ERRORLEVEL%"
call :clear
if not "%E1%"=="0" set "RC=%E1%"
if not "%E2%"=="0" set "RC=%E2%"
if not "%E3%"=="0" set "RC=%E3%"
echo ### tile %N%: %T%, exit test %E1% %E2% %E3% perf %E4% %E5% >> "%SUM%"
findstr /C:"large int8 tile" /C:"ignored" /C:"tests passed" /C:"FAIL" "%TLOG%" >> "%SUM%"
findstr /C:"MUL_MAT(" /C:"MUL_MAT_ID(" "%PLOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
