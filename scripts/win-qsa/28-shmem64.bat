@echo off
rem 64 KB of shared memory in vulkan on the AMD driver under windows. the driver reports 32 KB per
rem workgroup (maxComputeSharedMemorySize), RDNA has 64 KB, and HIP gives 64 KB on the same chip.
rem GGML_VK_SHMEM_LIMIT=65536 lets the shaders declare up to 64 KB. that is outside the Vulkan spec, so
rem the driver may refuse a shader, run it, compute garbage or hang the gpu - this is the check which one.
rem what the larger limit opens: the large int8 coopmat tile for the dense matmuls (512 threads, 128x128,
rem about 41 KB; GGML_VK_INT_LARGE_TILE=1 turns the large int8 tiles on, at 32 KB only matmul_id takes
rem them), larger flash attention tiles where their size follows the shared memory, more rows per pass in
rem solve_tri. arms: default (32 KB), 64k (the limit only), 64k-large (the limit and the large int8 tiles).
rem   1 64k-large against the CPU: MUL_MAT, MUL_MAT_ID, MUL_MAT_ID_FUSION, FLASH_ATTN_EXT (head size 128,
rem     and the sparse cases of 01), SOLVE_TRI; a FAIL or a crash stops before the model is loaded
rem   2 perf of the qwen4exp matmul shapes at 4096 tokens, every arm
rem   3 with the model: llama-bench pp4096 (ub 2048 and 4096), every arm
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 28-shmem64 is vulkan only
    exit /b 0
)

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

echo.
echo this run lets the shaders declare up to 64 KB of shared memory where the driver reports 32 KB.
echo that is outside the Vulkan spec: the driver may refuse the shaders, compute garbage or hang the
echo gpu. on a hang windows resets the driver: the screen goes dark for a moment and other programs
echo that use the gpu may lose it. save your work first.
echo.
rem run-all.bat has already decided, do not stop there
if not defined QSA_UNATTENDED pause

set "SUM=%LOGS%\28-shmem64-%TS%-summary.txt"
rem the quants with an int8 pipeline on RDNA3
set "TQP=type_a=(q4_0|q4_1|q5_0|q5_1|q8_0|iq4_nl|iq4_xs|mxfp4|q3_K|q4_K|q5_K|q6_K),"
rem the perf cases of 25-tile-sweep.bat: the q8_0 projections and the experts at 4096 tokens
set "MMP=type_a=q8_0,type_b=f32,m=(320|640|2560|6144|10240|12288),n=4096,k=(320|2560|6144|10240),bs=\[1,1\]"
set "IDP=n_mats=512,n_used=10,b=0,m=(640|2560),n=4096"
set "RC=0"
set "TESTRC=0"
set "TN=0"
call :armclear
echo writing %SUM%

call :test MUL_MAT "%TQP%"
call :test MUL_MAT_ID "%TQP%"
call :test MUL_MAT_ID_FUSION "%TQP%"
call :test FLASH_ATTN_EXT "hsk=128,hsv=128"
call :test FLASH_ATTN_EXT "n_kv_max=[1-9]"
call :test SOLVE_TRI ""
if not "%TESTRC%"=="0" goto :failed

for %%a in (default 64k 64k-large) do call :perf %%a

rem paths stay out of parenthesized blocks, a ")" in one would end the block
if not exist "%MODEL%" goto :nomodel
if not exist "%BIN%\llama-bench.exe" goto :nobench
for %%a in (default 64k 64k-large) do call :bench %%a
goto :end

:failed
echo ### a test failed or crashed with 64 KB, perf and model runs skipped >> "%SUM%"
goto :end

:nomodel
echo ### model not found: %MODEL%, the model runs are skipped >> "%SUM%"
goto :end

:nobench
echo ### not built: %BIN%\llama-bench.exe, the model runs are skipped >> "%SUM%"

:end
call :armclear

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = default, 64k or 64k-large
:armenv
call :armclear
if /i "%~1"=="64k" set "GGML_VK_SHMEM_LIMIT=65536"
if /i "%~1"=="64k-large" set "GGML_VK_SHMEM_LIMIT=65536"
if /i "%~1"=="64k-large" set "GGML_VK_INT_LARGE_TILE=1"
goto :eof

:armclear
set "GGML_VK_SHMEM_LIMIT="
set "GGML_VK_INT_LARGE_TILE="
goto :eof

rem %1 = the op, %2 = the case filter (in quotes, may be empty); always the 64k-large arm
:test
set /a TN+=1
set "LOG=%LOGS%\28-shmem64-%TS%-test-%TN%-%~1.log"
call :armenv 64k-large
rem the filter holds | and ( ), so it stays in quotes everywhere, and no parenthesized block
echo === test %~1 "%~2", 64k-large -^> %LOG%
if "%~2"=="" goto :test_nofilter
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o %~1 -p "%~2" > "%LOG%" 2>&1
goto :test_ran
:test_nofilter
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o %~1 > "%LOG%" 2>&1
:test_ran
set "EC=%ERRORLEVEL%"
call :armclear
findstr /C:"tests passed" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" set "RC=%EC%"
if not "%EC%"=="0" set "TESTRC=%EC%"
echo ### test %~1 "%~2", 64k-large, exit=%EC% >> "%SUM%"
findstr /C:"GGML_VK_SHMEM_LIMIT" /C:"large int8" /C:"tests passed" /C:"FAIL" /C:"rror" /C:"lost" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem %1 = an arm
:perf
set "LOG=%LOGS%\28-shmem64-%TS%-perf-%~1.log"
call :armenv %1
echo === perf, %~1 -^> %LOG%
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT -p "%MMP%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT_ID -p "%IDP%" >> "%LOG%" 2>&1
set "EC2=%ERRORLEVEL%"
call :armclear
if not "%EC%"=="0" set "RC=%EC%"
if not "%EC2%"=="0" set "RC=%EC2%"
echo ### perf, %~1, exit=%EC% %EC2% >> "%SUM%"
findstr /C:"MUL_MAT(" /C:"MUL_MAT_ID(" "%LOG%" >> "%SUM%"
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
set "LOG=%LOGS%\28-shmem64-%TS%-bench-%~1.log"
call :armenv %1
echo === llama-bench pp4096, %~1, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 0 -b 4096 -ub 2048,4096 -d 0 %LOADMODE% -r 2 %EXTRA% -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :armclear
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :bench_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### llama-bench pp4096, %~1, try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:" pp4096 " /C:"failed" /C:"rror" /C:"lost" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
