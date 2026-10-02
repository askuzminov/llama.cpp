@echo off
rem the fused expert kernel (FUSED_GLU, 02.10): the experts' up and gate projections and the swiglu over both
rem run as one int8 coopmat matmul_id with two A matrices. the B tile, the row ids and the launch are shared, and
rem the gate and up results never go to memory: the shader writes silu(gate) * up. GGML_VK_DISABLE_MMID_GLU=1
rem turns it off for the comparison. the log also shows the 64 KB shared memory check of the startup.
rem   1 MUL_MAT_VEC_FUSION (the up, gate, swiglu graph; m 16, 64, 200 tokens take the fused shader), MUL_MAT_ID
rem     and MUL_MAT_ID_FUSION against the CPU; a FAIL stops before the model is loaded
rem   2 with the model: llama-bench pp4096 (ub 2048 and 4096) and tg64, fused and not
rem   3 per-op profile of one ubatch of 4096 (GGML_VK_PERF_LOGGER), fused and not
rem   4 KLD of the fused path against the unfused int8 path on wikitext (INT8CTX x INT8CHUNKS tokens): the
rem     same arithmetic, so the result should sit at the floor
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 30-moe-glu is vulkan only
    exit /b 0
)

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

set "SUM=%LOGS%\30-moe-glu-%TS%-summary.txt"
rem the quants with an int8 pipeline on RDNA3
set "TQP=type_a=(q4_0|q4_1|q5_0|q5_1|q8_0|iq4_nl|iq4_xs|mxfp4|q3_K|q4_K|q5_K|q6_K),"
set "RC=0"
set "TESTRC=0"
set "GGML_VK_DISABLE_MMID_GLU="
set "GGML_VK_PERF_LOGGER="
echo writing %SUM%

call :test MUL_MAT_VEC_FUSION ""
call :test MUL_MAT_ID "%TQP%"
call :test MUL_MAT_ID_FUSION "%TQP%"
if not "%TESTRC%"=="0" goto :failed

rem paths stay out of parenthesized blocks, a ")" in one would end the block
if not exist "%MODEL%" goto :nomodel
if not exist "%BIN%\llama-bench.exe" goto :nomodel
rem 0 = fused (the default), 1 = GGML_VK_DISABLE_MMID_GLU
for %%f in (0 1) do call :bench %%f
for %%f in (0 1) do call :prof %%f
if not exist "%BIN%\llama-perplexity.exe" goto :end
if not exist "%PPLFILE%" goto :nowiki
call :kld
goto :end

:failed
echo ### a test failed, the model runs are skipped >> "%SUM%"
goto :end

:nomodel
echo ### model or llama-bench not found, the model runs are skipped >> "%SUM%"
goto :end

:nowiki
echo ### missing %PPLFILE%, run get-wikitext.bat for the KLD step >> "%SUM%"

:end
set "GGML_VK_DISABLE_MMID_GLU="
set "GGML_VK_PERF_LOGGER="

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = the op, %2 = the case filter in quotes (may be empty). the filter holds | and ( ), so it stays in
rem quotes and out of parenthesized blocks
:test
set "LOG=%LOGS%\30-moe-glu-%TS%-test-%~1.log"
echo === test %~1 "%~2" -^> %LOG%
if "%~2"=="" goto :test_nofilter
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o %~1 -p "%~2" > "%LOG%" 2>&1
goto :test_ran
:test_nofilter
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o %~1 > "%LOG%" 2>&1
:test_ran
set "EC=%ERRORLEVEL%"
findstr /C:"tests passed" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" set "RC=%EC%"
if not "%EC%"=="0" set "TESTRC=%EC%"
echo ### test %~1 "%~2", exit=%EC% >> "%SUM%"
findstr /C:"shared memory" /C:"int8 coopmat" /C:"tests passed" /C:"FAIL" "%LOG%" >> "%SUM%"
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
set "LOG=%LOGS%\30-moe-glu-%TS%-bench-%~1.log"
set "GGML_VK_DISABLE_MMID_GLU="
if "%~1"=="1" set "GGML_VK_DISABLE_MMID_GLU=1"
echo === llama-bench pp4096 tg64, fusion off %~1, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 64 -b 4096 -ub 2048,4096 -d 0 %LOADMODE% -r 2 %EXTRA% -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_DISABLE_MMID_GLU="
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :bench_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### llama-bench pp4096 tg64, fusion off %~1, try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:" pp4096 " /C:" tg64 " /C:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem the whole per-op table goes to the summary
:prof
set "TRY=0"
:prof_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\30-moe-glu-%TS%-prof-%~1.log"
set "GGML_VK_DISABLE_MMID_GLU="
if "%~1"=="1" set "GGML_VK_DISABLE_MMID_GLU=1"
set "GGML_VK_PERF_LOGGER=1"
echo === per-op profile pp4096 ub4096, fusion off %~1, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 0 -b 4096 -ub 4096 -d 0 %LOADMODE% -r 1 --no-warmup %EXTRA% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_DISABLE_MMID_GLU="
set "GGML_VK_PERF_LOGGER="
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :prof_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### profile pp4096 ub4096, fusion off %~1, try %TRY%, exit=%EC% >> "%SUM%"
findstr /R /C:" x .* us = " /C:"Total time" /C:" pp4096 " /C:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem base: the unfused int8 path; arm: the fused one
:kld
set "BASEFILE=%LOGS%\30-moe-glu-kl-base-%TS%.dat"
set "TRY=0"
:kld_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\30-moe-glu-%TS%-kl-base.log"
set "GGML_VK_DISABLE_MMID_GLU=1"
echo === kl base, fusion off, c=%INT8CTX% chunks=%INT8CHUNKS%, try %TRY% -^> %LOG%
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c %INT8CTX% --chunks %INT8CHUNKS% -fa on %LOADMODE% %EXTRA% --kl-divergence-base "%BASEFILE%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_DISABLE_MMID_GLU="
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :kld_try
findstr /c:"Final estimate" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" goto :klbasefail
set "TRY=0"
:klarm_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\30-moe-glu-%TS%-kl-fused.log"
echo === kl, fused against the unfused base, try %TRY% -^> %LOG%
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c %INT8CTX% --chunks %INT8CHUNKS% -fa on %LOADMODE% %EXTRA% --kl-divergence --kl-divergence-base "%BASEFILE%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :klarm_try
rem llama-perplexity returns 0 even when kl_divergence gives up, so check the numbers are there
findstr /c:"Mean    KLD" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" set "RC=%EC%"
echo ### kl, fused against the unfused int8 base, try %TRY%, exit=%EC% >> "%SUM%"
findstr /c:"Same top" /c:"Mean PPL(Q)/PPL(base)" /c:"Mean    KLD" /c:"RMS" /c:"failed" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
del "%BASEFILE%"
goto :eof

:klbasefail
set "RC=1"
echo ### kl base run failed, see %LOG% >> "%SUM%"
if exist "%BASEFILE%" del "%BASEFILE%"
goto :eof
