@echo off
rem column parts of matmuls with a small A and a large K (GGML_VK_MM_CHUNK_MB): when A and B do not fit the
rem cache budget, the columns run in parts one after another, so that A and the part of B in use stay in the
rem cache. the shape it is for is the qwen4exp hyper-connection down projection, q8_0 m 320 k 10240. on strix
rem halo with coopmat the q8_0 matmul takes B as f16 (there is no int8 pipeline next to coopmat), so int dot
rem changes nothing here. first the correctness with a 0.5 MB budget, so that small shapes are split too, then
rem the throughput grid over the budget: off, 16, 24 (the default on an AMD APU) and 32 MB. no model needed.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 24-mmchunk is vulkan only
    exit /b 0
)

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

set "SUM=%LOGS%\24-mmchunk-%TS%-summary.txt"
rem the shapes of the perf cases in test-backend-ops: the down projection at four ubatch sizes and neighbours
set "MMP=type_a=q8_0,type_b=f32,m=(320|640|2560),n=(512|1024|2048|4096),k=(2560|10240),bs=\[1,1\]"
set "RC=0"
echo writing %SUM%

set "LOG=%LOGS%\24-mmchunk-%TS%-test.log"
set "GGML_VK_MM_CHUNK_MB=0.5"
echo === test, budget 0.5 MB -^> %LOG%
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o MUL_MAT > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo ### test, budget 0.5 MB, exit=%EC% >> "%SUM%"
findstr /C:"tests passed" /C:"FAIL" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"

rem budget in MB, 0 = off
for %%b in (0 16 24 32) do call :perf %%b

set "GGML_VK_MM_CHUNK_MB="

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

:perf
set "LOG=%LOGS%\24-mmchunk-%TS%-perf-%~1.log"
set "GGML_VK_MM_CHUNK_MB=%~1"
echo === perf, budget %~1 MB -^> %LOG%
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT -p "%MMP%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo ### perf, budget %~1 MB, exit=%EC% >> "%SUM%"
findstr /C:"MUL_MAT(" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
