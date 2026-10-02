@echo off
rem the int8 coopmat matmuls of the prefill (mul_mmq_cm1: activations quantized to q8_1, int8 WMMA) on
rem the AMD driver under windows. GGML_VK_INT_COOPMAT=1 turns them on, by default they run on RADV only.
rem the fork turned them off after upstream #29392 (garbage with the AMD driver on windows), but that
rem issue was an old driver that hid shader_float8, so RDNA4 ran with the RDNA3 accumulator layout; it is
rem closed. int8 WMMA is the route gufo takes for its dense q8_0 and expert matmuls. first MUL_MAT,
rem MUL_MAT_ID and MUL_MAT_ID_FUSION (the router weight applied as the matmul writes) against the CPU for
rem the quants with an int8 pipeline; a FAIL ends the run before the model is loaded. then the
rem throughput of the qwen4exp shapes at 4096 tokens, int8 off and on. then, when the model is there:
rem llama-bench pp4096 (ub 2048 and 4096) and tg64, and the KLD against the f16 path on wikitext, for
rem every value of GGML_VK_INT_COOPMAT: 0 off (the KLD floor), 1 MUL_MAT and MUL_MAT_ID, 2 MUL_MAT only,
rem 3 MUL_MAT_ID only. "26-int8.bat kld" runs the KLD step only
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 26-int8 is vulkan only
    exit /b 0
)

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

set "SUM=%LOGS%\26-int8-%TS%-summary.txt"
rem the quants with an int8 pipeline on RDNA3
set "TQP=type_a=(q4_0|q4_1|q5_0|q5_1|q8_0|iq4_nl|iq4_xs|mxfp4|q3_K|q4_K|q5_K|q6_K),"
rem the perf cases of 25-tile-sweep.bat: the q8_0 projections and the experts at 4096 tokens
set "MMP=type_a=q8_0,type_b=f32,m=(320|640|2560|6144|10240|12288),n=4096,k=(320|2560|6144|10240),bs=\[1,1\]"
set "IDP=n_mats=512,n_used=10,b=0,m=(640|2560),n=4096"
set "RC=0"
set "TESTRC=0"
echo writing %SUM%

if /i "%~1"=="kld" goto :kldonly

call :test MUL_MAT
call :test MUL_MAT_ID
call :test MUL_MAT_ID_FUSION
if not "%TESTRC%"=="0" goto :failed

for %%i in (0 1) do call :perf %%i

rem paths stay out of parenthesized blocks, a ")" in one would end the block
if not exist "%MODEL%" goto :nomodel
if not exist "%BIN%\llama-bench.exe" goto :nobench
for %%i in (0 1 2 3) do call :bench %%i

:kldonly
if not exist "%MODEL%" goto :nomodel
if not exist "%BIN%\llama-perplexity.exe" goto :end
if not exist "%PPLFILE%" goto :nowiki
call :kld
goto :end

:failed
echo ### a test failed: the int8 path gives wrong results here, perf and model runs skipped >> "%SUM%"
goto :end

:nomodel
echo ### model not found: %MODEL%, the model runs are skipped >> "%SUM%"
goto :end

:nobench
echo ### not built: %BIN%\llama-bench.exe, the model runs are skipped >> "%SUM%"
goto :end

:nowiki
echo ### missing %PPLFILE%, run get-wikitext.bat for the KLD step >> "%SUM%"

:end
set "GGML_VK_INT_COOPMAT="

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = the op to test, always with int8 on
:test
set "LOG=%LOGS%\26-int8-%TS%-test-%~1.log"
set "GGML_VK_INT_COOPMAT=1"
echo === test %~1, int8 on -^> %LOG%
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o %~1 -p "%TQP%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_INT_COOPMAT="
findstr /C:"tests passed" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" set "RC=%EC%"
if not "%EC%"=="0" set "TESTRC=%EC%"
echo ### test %~1, int8 on, exit=%EC% >> "%SUM%"
findstr /C:"int8 coopmat" /C:"tests passed" /C:"FAIL" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem %1 = 0 int8 off (the f16 path), 1 int8 on
:perf
set "LOG=%LOGS%\26-int8-%TS%-perf-%~1.log"
set "GGML_VK_INT_COOPMAT=%~1"
echo === perf, int8 %~1 -^> %LOG%
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT -p "%MMP%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT_ID -p "%IDP%" >> "%LOG%" 2>&1
set "EC2=%ERRORLEVEL%"
set "GGML_VK_INT_COOPMAT="
if not "%EC%"=="0" set "RC=%EC%"
if not "%EC2%"=="0" set "RC=%EC2%"
echo ### perf, int8 %~1, exit=%EC% %EC2% >> "%SUM%"
findstr /C:"MUL_MAT(" /C:"MUL_MAT_ID(" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem the driver frees the memory of the previous process lazily, and a model load right after it
rem can fail to allocate even after SETTLE (README, traps). every model run below gets one more
rem try after another wait when its log says "failed to allocate"
:bench
set "TRY=0"
:bench_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\26-int8-%TS%-bench-%~1.log"
set "GGML_VK_INT_COOPMAT=%~1"
echo === llama-bench pp4096 tg64, int8 %~1, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 64 -b 4096 -ub 2048,4096 -d 0 %LOADMODE% -r 2 %EXTRA% -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_INT_COOPMAT="
findstr /C:"failed to allocate" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :bench_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### llama-bench pp4096 tg64, int8 %~1, try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:" pp4096 " /C:" tg64 " /C:"int8 coopmat" /C:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem the base logits come from the f16 path (int8 off). arm 0 repeats the base and gives the noise
rem floor, arms 1-3 are the int8 scopes
:kld
set "BASEFILE=%LOGS%\26-int8-kl-base-%TS%.dat"
set "TRY=0"
:kld_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\26-int8-%TS%-kl-base.log"
set "GGML_VK_INT_COOPMAT=0"
echo === kl base, int8 off, c=%INT8CTX% chunks=%INT8CHUNKS%, try %TRY% -^> %LOG%
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c %INT8CTX% --chunks %INT8CHUNKS% -fa on %LOADMODE% %EXTRA% --kl-divergence-base "%BASEFILE%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_INT_COOPMAT="
findstr /C:"failed to allocate" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :kld_try
rem a crash leaves a truncated .dat; the base run prints "Final estimate" as its last line
findstr /c:"Final estimate" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" goto :klbasefail
for %%i in (0 1 2 3) do call :klarm %%i
del "%BASEFILE%"
goto :eof

:klbasefail
set "RC=1"
echo ### kl base run failed, see %LOG% >> "%SUM%"
if exist "%BASEFILE%" del "%BASEFILE%"
goto :eof

:klarm
set "TRY=0"
:klarm_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\26-int8-%TS%-kl-%~1.log"
set "GGML_VK_INT_COOPMAT=%~1"
echo === kl, int8 %~1 against the f16 base, try %TRY% -^> %LOG%
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c %INT8CTX% --chunks %INT8CHUNKS% -fa on %LOADMODE% %EXTRA% --kl-divergence --kl-divergence-base "%BASEFILE%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_INT_COOPMAT="
findstr /C:"failed to allocate" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :klarm_try
rem llama-perplexity returns 0 even when kl_divergence gives up, so check the numbers are there
findstr /c:"Mean    KLD" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" set "RC=%EC%"
echo ### kl, int8 %~1 against the f16 base, try %TRY%, exit=%EC% >> "%SUM%"
findstr /c:"Same top" /c:"Mean PPL(Q)/PPL(base)" /c:"Mean    KLD" /c:"RMS" /c:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
