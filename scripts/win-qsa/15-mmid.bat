@echo off
rem MUL_MAT_ID scale epilogue, one model load per arm. MUL_MAT_ID is the largest single op of
rem the prefill graph after flash attention, so this only matters at a large ubatch.
rem the following MUL (ffn_moe_weighted) is applied as the matmul writes out, which drops a full
rem write and read back of the matmul result. on by default, refused on coopmat2.
rem   GGML_VK_MMID_SCALE_EPILOGUE_DISABLE   back to the plain matmul plus a separate MUL
rem the epilogue changes what is written, so the backend test runs first; the result must stay
rem bit-comparable against the cpu reference.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem the epilogue is a vulkan pipeline, another backend has nothing to switch
if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 15-mmid is vulkan only
    exit /b 0
)

rem arms: on default (epilogue), off disabled
if not defined MMIDVARIANTS set "MMIDVARIANTS=on off"

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

set "SUM=%LOGS%\15-mmid-%TS%-summary.txt"
set "RC=0"

rem correctness of the fused epilogue, no model needed
if exist "%BIN%\test-backend-ops.exe" (
    set "LOG=%LOGS%\15-mmid-%TS%-test.log"
    echo === test-backend-ops MUL_MAT_ID_FUSION -^> !LOG!
    set "GGML_VK_MMID_SCALE_EPILOGUE_DISABLE="
    "%BIN%\test-backend-ops.exe" test -o MUL_MAT_ID_FUSION > "!LOG!" 2>&1
    set "EC=!ERRORLEVEL!"
    "%BIN%\test-backend-ops.exe" test -o MUL_MAT_ID >> "!LOG!" 2>&1
    if not "!ERRORLEVEL!"=="0" set "EC=!ERRORLEVEL!"
    if not "!EC!"=="0" set "RC=!EC!"
    echo ### test-backend-ops exit=!EC! >> "%SUM%"
    type "!LOG!" >> "%SUM%"

    rem qwen4exp MUL_MAT_ID throughput: q4_K gate/up against q4_0 at the same shape and the same
    rem bytes per weight, plus the q5_1 down projection. coopmat1 runs every quant on the same
    rem warptile, so the difference between the arms is the dequant of the type.
    set "PLOG=%LOGS%\15-mmid-%TS%-perf.log"
    echo === test-backend-ops perf MUL_MAT_ID n_used=10 -^> !PLOG!
    "%BIN%\test-backend-ops.exe" perf -o MUL_MAT_ID -p n_used=10 > "!PLOG!" 2>&1
    echo. >> "%SUM%"
    echo ### test-backend-ops perf >> "%SUM%"
    type "!PLOG!" >> "%SUM%"
)

for %%v in (%MMIDVARIANTS%) do call :run %%v

set "GGML_VK_MMID_SCALE_EPILOGUE_DISABLE="

echo.
echo summary in %SUM%
type "%SUM%"
exit /b %RC%

rem %1 = arm: on default, off epilogue disabled
:run
set "LOG=%LOGS%\15-mmid-%TS%-%~1.log"
set "GGML_VK_MMID_SCALE_EPILOGUE_DISABLE="
if "%~1"=="off" set "GGML_VK_MMID_SCALE_EPILOGUE_DISABLE=1"
echo === mmid=%~1 -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p %NPROMPT% -n 0 -b 4096 -ub %UBATCH% -d %DEPTHS% %LOADMODE% -r %REPS% %EXTRA% --progress -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo. >> "%SUM%"
echo ### mmid=%~1 exit=%EC% >> "%SUM%"
type "%LOG%" >> "%SUM%"
goto :eof
