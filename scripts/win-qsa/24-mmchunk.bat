@echo off
rem column parts of big-B matmuls (GGML_VK_MM_CHUNK_MB): a matmul whose staged B is larger than the cache
rem budget runs its columns in parts, so the strips of M do not read B from memory again for every strip.
rem the shape it is for is the qwen4exp hyper-connection down projection, q8_0 m 320 k 10240. first the
rem correctness with a 0.25 MB budget, so that small shapes are split too, on the int8 path (B quantized to
rem q8_1, the default on strix halo) and on the coopmat f16 path (int dot off). then the throughput grid:
rem both paths, with the default budget (16 MB on an AMD APU) and with the parts off. no model needed.
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

rem arm name, int dot off (1) or on (0)
call :test mmq 0
call :test cm 1

rem arm name, int dot off (1) or on (0), budget in MB (empty = the default)
call :perf mmq-16 0 ""
call :perf mmq-0 0 0
call :perf cm-16 1 ""
call :perf cm-0 1 0

set "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT="
set "GGML_VK_MM_CHUNK_MB="

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

:test
set "LOG=%LOGS%\24-mmchunk-%TS%-test-%~1.log"
set "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT="
if "%~2"=="1" set "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1"
set "GGML_VK_MM_CHUNK_MB=0.25"
echo === test %~1 -^> %LOG%
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o MUL_MAT > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo ### test %~1, budget 0.25 MB, exit=%EC% >> "%SUM%"
findstr /C:"tests passed" /C:"FAIL" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

:perf
set "LOG=%LOGS%\24-mmchunk-%TS%-perf-%~1.log"
set "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT="
if "%~2"=="1" set "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1"
set "GGML_VK_MM_CHUNK_MB=%~3"
echo === perf %~1 -^> %LOG%
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT -p "%MMP%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo ### perf %~1, exit=%EC% >> "%SUM%"
findstr /C:"MUL_MAT(" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
