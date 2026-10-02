@echo off
rem the int8 coopmat matmuls of the prefill (mul_mmq_cm1: activations quantized to q8_1, int8 WMMA) and the
rem f32 accumulators of the f16 path, the default on the AMD driver for RDNA3 since 02.10.2026, against the
rem f16 path (GGML_VK_INT_COOPMAT=0) and the old default (f16 path with f16 accumulators, GGML_VK_F16ACC=1).
rem first MUL_MAT, MUL_MAT_ID and MUL_MAT_ID_FUSION with the default against the CPU for the quants with an
rem int8 pipeline (a FAIL ends the run before the model is loaded), then the throughput of the qwen4exp shapes
rem at 4096 tokens, then with the model llama-bench pp4096 (ub 2048 and 4096) and tg64, and the KLD on
rem wikitext against the most exact base: the f16 path with f32 accumulators. arms of the KLD: the default,
rem the old default, and the base with -ub 256 (only the order of the sums changes, the noise floor of this
rem model). "26-int8.bat kld" runs the KLD step only. "26-int8.bat kldlong" runs it on INT8LONGCHUNKS
rem chunks instead of INT8CHUNKS (32 by default: 8 times the tokens, a base logits file of about 16 GB),
rem with the same arms: per-chunk values in the logs allow a paired comparison of the arms
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
set "KN=0"
set "BN=0"
call :armclear
echo writing %SUM%

rem an arm is "MODE EXTRA": MODE is default (no knob) or the value of GGML_VK_INT_COOPMAT (0 off, 1 on,
rem 2 MUL_MAT only, 3 MUL_MAT_ID only), EXTRA is empty or acc16, acc32, ub256, acc32ub256, see :armenv
set "BENCHARMS="default" "0" "0 acc16""
set "KLARMS="default" "0 acc16" "0 acc32ub256""
set "BASEARM=0 acc32"
if /i "%~1"=="kld" goto :kldonly
if /i "%~1"=="kldlong" set "INT8CHUNKS=%INT8LONGCHUNKS%"
if /i "%~1"=="kldlong" goto :kldonly

call :test MUL_MAT
call :test MUL_MAT_ID
call :test MUL_MAT_ID_FUSION
if not "%TESTRC%"=="0" goto :failed

for %%i in (0 default) do call :perf %%i

rem paths stay out of parenthesized blocks, a ")" in one would end the block
if not exist "%MODEL%" goto :nomodel
if not exist "%BIN%\llama-bench.exe" goto :nobench
for %%i in (%BENCHARMS%) do call :bench %%i

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
call :armclear

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = an arm, see above. sets the knobs, ARGS (extra arguments) and ARMLABEL
:armenv
set "M="
set "X="
for /f "tokens=1,2" %%a in ("%~1") do (
    set "M=%%a"
    set "X=%%b"
)
call :armclear
set "ARGS="
if not "%M%"=="default" set "GGML_VK_INT_COOPMAT=%M%"
if "%X%"=="acc16" set "GGML_VK_F16ACC=1"
if "%X%"=="acc32" set "GGML_VK_F16ACC=0"
if "%X%"=="ub256" set "ARGS=-ub 256"
if "%X%"=="acc32ub256" set "GGML_VK_F16ACC=0"
if "%X%"=="acc32ub256" set "ARGS=-ub 256"
set "ARMLABEL=int8 %M% %X%"
goto :eof

:armclear
set "GGML_VK_INT_COOPMAT="
set "GGML_VK_F16ACC="
goto :eof

rem %1 = the op to test, with the default
:test
set "LOG=%LOGS%\26-int8-%TS%-test-%~1.log"
call :armclear
echo === test %~1, default -^> %LOG%
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o %~1 -p "%TQP%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
findstr /C:"tests passed" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" set "RC=%EC%"
if not "%EC%"=="0" set "TESTRC=%EC%"
echo ### test %~1, default, exit=%EC% >> "%SUM%"
findstr /C:"int8 coopmat" /C:"tests passed" /C:"FAIL" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem %1 = an arm
:perf
set "LOG=%LOGS%\26-int8-%TS%-perf-%~1.log"
call :armenv %1
echo === perf, %ARMLABEL% -^> %LOG%
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT -p "%MMP%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT_ID -p "%IDP%" >> "%LOG%" 2>&1
set "EC2=%ERRORLEVEL%"
call :armclear
if not "%EC%"=="0" set "RC=%EC%"
if not "%EC2%"=="0" set "RC=%EC2%"
echo ### perf, %ARMLABEL%, exit=%EC% %EC2% >> "%SUM%"
findstr /C:"MUL_MAT(" /C:"MUL_MAT_ID(" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem the driver frees the memory of the previous process lazily, and a model load right after it
rem can fail to allocate even after SETTLE (README, traps). every model run below gets one more
rem try after another wait when its log says "failed to allocate" or "Device memory allocation"
:bench
set /a BN+=1
set "TRY=0"
:bench_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\26-int8-%TS%-bench-%BN%.log"
call :armenv %1
echo === llama-bench pp4096 tg64 %BN%, %ARMLABEL%, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 64 -b 4096 -ub 2048,4096 -d 0 %LOADMODE% -r 2 %EXTRA% -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :armclear
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :bench_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### llama-bench pp4096 tg64 %BN%, %ARMLABEL%, try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:" pp4096 " /C:" tg64 " /C:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem the base logits come from BASEARM: the f16 path with f32 accumulators
:kld
set "BASEFILE=%LOGS%\26-int8-kl-base-%TS%.dat"
set "TRY=0"
:kld_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\26-int8-%TS%-kl-base.log"
call :armenv "%BASEARM%"
echo === kl base, %ARMLABEL%, c=%INT8CTX% chunks=%INT8CHUNKS%, try %TRY% -^> %LOG%
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c %INT8CTX% --chunks %INT8CHUNKS% -fa on %LOADMODE% %EXTRA% %ARGS% --kl-divergence-base "%BASEFILE%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :armclear
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :kld_try
rem a crash leaves a truncated .dat; the base run prints "Final estimate" as its last line
findstr /c:"Final estimate" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" goto :klbasefail
echo ### kl base: %ARMLABEL% >> "%SUM%"
findstr /c:"Final estimate" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
for %%i in (%KLARMS%) do call :klarm %%i
del "%BASEFILE%"
goto :eof

:klbasefail
set "RC=1"
echo ### kl base run failed, see %LOG% >> "%SUM%"
if exist "%BASEFILE%" del "%BASEFILE%"
goto :eof

:klarm
set /a KN+=1
set "TRY=0"
:klarm_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\26-int8-%TS%-kl-%KN%.log"
call :armenv %1
echo === kl %KN%, %ARMLABEL% against the base, try %TRY% -^> %LOG%
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c %INT8CTX% --chunks %INT8CHUNKS% -fa on %LOADMODE% %EXTRA% %ARGS% --kl-divergence --kl-divergence-base "%BASEFILE%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :armclear
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :klarm_try
rem llama-perplexity returns 0 even when kl_divergence gives up, so check the numbers are there
findstr /c:"Mean    KLD" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" set "RC=%EC%"
echo ### kl %KN%, %ARMLABEL% against the base, try %TRY%, exit=%EC% >> "%SUM%"
findstr /c:"Same top" /c:"Mean PPL(Q)/PPL(base)" /c:"Mean    KLD" /c:"RMS" /c:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
