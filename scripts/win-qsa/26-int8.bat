@echo off
rem the int8 coopmat matmuls of the prefill (mul_mmq_cm1: activations quantized to q8_1, int8 WMMA) on
rem the AMD driver under windows. GGML_VK_INT_COOPMAT=1 turns them on, by default they run on RADV only.
rem the fork turned them off after upstream #29392 (garbage with the AMD driver on windows), but that
rem issue was an old driver that hid shader_float8, so RDNA4 ran with the RDNA3 accumulator layout; it is
rem closed. int8 WMMA is the route gufo takes for its dense q8_0 and expert matmuls.
rem   26-int8.bat         MUL_MAT, MUL_MAT_ID and MUL_MAT_ID_FUSION against the CPU for the quants with an
rem                       int8 pipeline (a FAIL ends the run before the model is loaded), the throughput of
rem                       the qwen4exp shapes at 4096 tokens, then with the model llama-bench pp4096 (ub 2048
rem                       and 4096) and tg64 and the KLD against the f16 path on wikitext for every value of
rem                       GGML_VK_INT_COOPMAT: 0 off (the KLD floor of the same code), 1 MUL_MAT and
rem                       MUL_MAT_ID, 2 MUL_MAT only, 3 MUL_MAT_ID only
rem   26-int8.bat kld     the KLD step only
rem   26-int8.bat bisect  the KLD of single weight groups on int8 (GGML_VK_INT_COOPMAT_FILTER)
rem   26-int8.bat floor   how far the f16 path itself moves when only the order of the sums changes
rem                       (-ub 256, fusion off, the scalar path: no coopmat, no integer dot), next to int8
rem                       without the hc mixers and int8 everywhere; plus llama-bench of those int8 arms
rem   26-int8.bat ref     the KLD against a more exact base: the f16 path with f32 accumulators
rem                       (GGML_VK_DISABLE_F16ACC). the default f16 path sums quantized matmuls in f16
rem                       accumulators when the device has them, the int8 path in int32 and f32: which one
rem                       is nearer to the exact sums. arms: the default, int8 everywhere, int8 without the
rem                       hc mixers, the base with -ub 256 (the noise floor of a reordering at the same
rem                       precision) and f32 everywhere (GGML_VK_DISABLE_F16); plus llama-bench of the
rem                       default, of f32 accumulators and of int8 with f32 accumulators
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

rem an arm is "MODE FILTER EXTRA": MODE is GGML_VK_INT_COOPMAT, FILTER is GGML_VK_INT_COOPMAT_FILTER
rem (parts of weight names, -x excludes, "-" alone or nothing is no filter), EXTRA is empty or ub256,
rem nofuse, scalar, acc32, acc32ub256, f32, see :armenv. BASEARM is the arm of the KLD base.
rem bisect: all but the hc mixers, then one group at a time - hc mixers, shared expert, attention and
rem GDN projections, PLE and indexer, expert down, expert gate and up ("attn_" also matches hc_attn_*,
rem hence the -hc_)
set "BENCHARMS="0" "1" "2" "3""
set "KLARMS="0" "1" "2" "3""
set "BASEARM=0"
if /i "%~1"=="bisect" set "KLARMS="1 -hc_" "2 hc_" "2 shexp" "2 attn_,ssm_,-hc_" "2 ple_,indexer" "3 down_exps" "3 gate_exps,up_exps""
if /i "%~1"=="floor" set "BENCHARMS="0" "1" "1 -hc_""
if /i "%~1"=="floor" set "KLARMS="0 - ub256" "0 - nofuse" "0 - scalar" "1 -hc_" "1""
if /i "%~1"=="ref" set "BENCHARMS="0" "0 - acc32" "1 - acc32""
if /i "%~1"=="ref" set "KLARMS="0" "1 - acc32" "1 -hc_ acc32" "0 - acc32ub256" "0 - f32""
if /i "%~1"=="ref" set "BASEARM=0 - acc32"
if /i "%~1"=="ref" goto :benchonly
if /i "%~1"=="bisect" goto :kldonly
if /i "%~1"=="kld" goto :kldonly
if /i "%~1"=="floor" goto :benchonly

call :test MUL_MAT
call :test MUL_MAT_ID
call :test MUL_MAT_ID_FUSION
if not "%TESTRC%"=="0" goto :failed

for %%i in (0 1) do call :perf %%i

:benchonly
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

rem %1 = an arm, see above. sets the knobs and ARGS (extra arguments) and ARMLABEL
:armenv
set "M="
set "F="
set "X="
for /f "tokens=1,2,3" %%a in ("%~1") do (
    set "M=%%a"
    set "F=%%b"
    set "X=%%c"
)
if "%F%"=="-" set "F="
call :armclear
set "ARGS="
set "GGML_VK_INT_COOPMAT=%M%"
set "GGML_VK_INT_COOPMAT_FILTER=%F%"
if "%X%"=="ub256" set "ARGS=-ub 256"
if "%X%"=="nofuse" set "GGML_VK_DISABLE_FUSION=1"
if "%X%"=="scalar" set "GGML_VK_DISABLE_COOPMAT=1"
if "%X%"=="scalar" set "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1"
if "%X%"=="acc32" set "GGML_VK_DISABLE_F16ACC=1"
if "%X%"=="acc32ub256" set "GGML_VK_DISABLE_F16ACC=1"
if "%X%"=="acc32ub256" set "ARGS=-ub 256"
if "%X%"=="f32" set "GGML_VK_DISABLE_F16=1"
set "ARMLABEL=int8 %M% filter "%F%" %X%"
goto :eof

:armclear
set "GGML_VK_INT_COOPMAT="
set "GGML_VK_INT_COOPMAT_FILTER="
set "GGML_VK_DISABLE_FUSION="
set "GGML_VK_DISABLE_COOPMAT="
set "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT="
set "GGML_VK_DISABLE_F16ACC="
set "GGML_VK_DISABLE_F16="
goto :eof

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
findstr /C:" pp4096 " /C:" tg64 " /C:"int8 coopmat" /C:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem the base logits come from BASEARM: the f16 path, or with f32 accumulators in the ref mode
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
findstr /c:"f16 accumulators" /c:"Final estimate" "%LOG%" >> "%SUM%"
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
echo === kl %KN%, %ARMLABEL% against the f16 base, try %TRY% -^> %LOG%
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c %INT8CTX% --chunks %INT8CHUNKS% -fa on %LOADMODE% %EXTRA% %ARGS% --kl-divergence --kl-divergence-base "%BASEFILE%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :armclear
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :klarm_try
rem llama-perplexity returns 0 even when kl_divergence gives up, so check the numbers are there
findstr /c:"Mean    KLD" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" set "RC=%EC%"
echo ### kl %KN%, %ARMLABEL% against the f16 base, try %TRY%, exit=%EC% >> "%SUM%"
findstr /c:"Same top" /c:"Mean PPL(Q)/PPL(base)" /c:"Mean    KLD" /c:"RMS" /c:"failed" /c:"int8 coopmat filter" /c:"f16 accumulators" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
