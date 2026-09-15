@echo off
rem MUL_MAT_ID tuning, two switches, one model load per arm. MUL_MAT_ID is the largest single
rem op of the prefill graph, so both switches only matter at a large ubatch.
rem   GGML_VK_MMID_WG256          the dense large tile runs 256 threads on a 128x128 tile,
rem                               the mul_mat_id tiles still run 128. this raises both the
rem                               medium and the large mul_mat_id tile to 256 threads
rem   GGML_VK_MMID_SCALE_EPILOGUE the following MUL (ffn_moe_weighted) is applied as the
rem                               matmul writes out, which drops a full write and read back
rem                               of the matmul result. refused on coopmat2
rem the epilogue changes what is written, so the arm that turns it on also runs the backend
rem test first; the result must stay bit-comparable against the cpu reference.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem arms: b base, w wg256, e scale epilogue, we both
if not defined MMIDVARIANTS set "MMIDVARIANTS=b w e we"

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
    set "GGML_VK_MMID_SCALE_EPILOGUE=1"
    set "GGML_VK_MMID_WG256=1"
    "%BIN%\test-backend-ops.exe" test -o MUL_MAT_ID_FUSION > "!LOG!" 2>&1
    set "EC=!ERRORLEVEL!"
    "%BIN%\test-backend-ops.exe" test -o MUL_MAT_ID >> "!LOG!" 2>&1
    if not "!ERRORLEVEL!"=="0" set "EC=!ERRORLEVEL!"
    set "GGML_VK_MMID_SCALE_EPILOGUE="
    set "GGML_VK_MMID_WG256="
    if not "!EC!"=="0" set "RC=!EC!"
    echo ### test-backend-ops exit=!EC! >> "%SUM%"
    type "!LOG!" >> "%SUM%"
)

for %%v in (%MMIDVARIANTS%) do call :run %%v

set "GGML_VK_MMID_WG256="
set "GGML_VK_MMID_SCALE_EPILOGUE="

echo.
echo summary in %SUM%
type "%SUM%"
exit /b %RC%

rem %1 = arm: b base, w wg256, e epilogue, we both
:run
set "LOG=%LOGS%\15-mmid-%TS%-%~1.log"
set "GGML_VK_MMID_WG256="
set "GGML_VK_MMID_SCALE_EPILOGUE="
if "%~1"=="w"  set "GGML_VK_MMID_WG256=1"
if "%~1"=="e"  set "GGML_VK_MMID_SCALE_EPILOGUE=1"
if "%~1"=="we" set "GGML_VK_MMID_WG256=1"
if "%~1"=="we" set "GGML_VK_MMID_SCALE_EPILOGUE=1"
echo === mmid=%~1 -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p %NPROMPT% -n 0 -b 4096 -ub %UBATCH% -d %DEPTHS% %LOADMODE% -r %REPS% %EXTRA% --progress -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo. >> "%SUM%"
echo ### mmid=%~1 exit=%EC% >> "%SUM%"
type "%LOG%" >> "%SUM%"
goto :eof
