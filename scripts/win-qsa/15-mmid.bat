@echo off
rem MUL_MAT_ID scale epilogue: the following MUL (ffn_moe_weighted) is applied as the matmul
rem writes out, which drops a full write and read back of the matmul result. always on, refused
rem on coopmat2. the epilogue changes what is written, so the result must stay bit-comparable
rem against the cpu reference. no model needed: 03-prefill measures the prefill with it.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem the epilogue is a vulkan pipeline, another backend has nothing to check
if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 15-mmid is vulkan only
    exit /b 0
)

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

set "SUM=%LOGS%\15-mmid-%TS%-summary.txt"
set "LOG=%LOGS%\15-mmid-%TS%-test.log"
set "PLOG=%LOGS%\15-mmid-%TS%-perf.log"

rem correctness of the fused epilogue
echo === test-backend-ops MUL_MAT_ID_FUSION -^> %LOG%
"%BIN%\test-backend-ops.exe" test -o MUL_MAT_ID_FUSION > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" test -o MUL_MAT_ID >> "%LOG%" 2>&1
if not "%ERRORLEVEL%"=="0" set "EC=%ERRORLEVEL%"
echo ### test-backend-ops exit=%EC% >> "%SUM%"
type "%LOG%" >> "%SUM%"

rem qwen4exp MUL_MAT_ID throughput: q4_K gate/up against q4_0 at the same shape and the same
rem bytes per weight, plus the q5_1 down projection. coopmat1 runs every quant on the same
rem warptile, so the difference between the arms is the dequant of the type.
echo === test-backend-ops perf MUL_MAT_ID n_used=10 -^> %PLOG%
"%BIN%\test-backend-ops.exe" perf -o MUL_MAT_ID -p n_used=10 > "%PLOG%" 2>&1
echo. >> "%SUM%"
echo ### test-backend-ops perf >> "%SUM%"
type "%PLOG%" >> "%SUM%"

echo.
echo summary in %SUM%
type "%SUM%"
exit /b %EC%
