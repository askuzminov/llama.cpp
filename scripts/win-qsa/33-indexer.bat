@echo off
rem the indexer and the host inputs at depth (03.10). the 32 profile of one 4096-token ubatch at d122880
rem (-c 262144) against d0: flash attention 912 ms (298), TOP_K 337 (12), MULTI_ADD 253 (114), RELU 209 (7),
rem the indexer score matmul 172 (7), and 1.17 s of host work per ubatch (0.22). the changes, each with a switch
rem back for the comparison:
rem   inputs: the KQ mask and the QSA block bias are filled on several threads, LLAMA_INPUT_THREADS (default
rem     half the cores, at most 8; 1 = one thread, as before). LLAMA_INPUT_TIMING=1 prints the host time of
rem     each ubatch on stderr: graph build, input fills, graph submission, and the fills of 0.1 ms or more
rem   head sum: the 4 heads of the indexer in one matmul (MUL_MAT_HEADSUM in the profile). it multiplies the block
rem     keys by the q of all heads, and the store takes the relu of each head, sums the heads of a token and adds
rem     the block bias: one 520 MB score write per layer at d122880 instead of 4 head scores, 4 relu passes, the
rem     sum and the bias add. in decode it reads the keys once instead of once per head.
rem     GGML_VK_DISABLE_MM_HEADSUM=1 turns it off
rem   relu: without the head sum, the relu of each head is done at the store of its matmul (MUL_MAT_RELU) and the
rem     head sum and the bias add are one MULTI_ADD. GGML_VK_DISABLE_MM_RELU=1 gives the separate RELU
rem   top-k: since the run of 03.10 09:48 the default takes radix-select for k >= 256 over 8 or more rows of 4096 or
rem     more (prefill: 31744 x 4096 k=513 5.0 against 19.3 ms there) and the tournament for fewer rows (decode, draft
rem     checks). GGML_VK_TOPK_RADIX=1 takes radix-select for every top-k, 0 the tournament where k fits
rem   1 tests against the CPU: MUL_MAT (the matmul push constants changed), MUL_MAT_RELU, MUL_MAT_HEADSUM (the
rem     coopmat store of the head sum runs only here, the Mac has no coopmat), TOP_K with both top-k paths. a FAIL
rem     stops before the model is loaded
rem   2 perf: TOP_K k=513 (1 to 4096 rows), tournament and radix. the fused matmuls have no perf step: perf mode
rem     of test-backend-ops repeats only the last node of a graph; the profile of step 4 has their times
rem   3 llama-bench pp4096 and tg32 at DEPTH33 with -c CTX33 for each arm of ARMS33, LLAMA_INPUT_TIMING on.
rem     the depth is filled once per arm, outside the timing
rem   4 per-op profile of one ubatch at DEPTH33 for the arms of PROF33 (GGML_VK_PERF_LOGGER, the last graph
rem     of the log is the measured one)
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 33-indexer is vulkan only
    exit /b 0
)

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

set "SUM=%LOGS%\33-indexer-%TS%-summary.txt"
set "RC=0"
set "TESTRC=0"
call :clearknobs
echo writing %SUM%
echo ### 33-indexer %TS%, depth %DEPTH33%, ctx %CTX33%, arms %ARMS33%, profiles %PROF33% > "%SUM%"
echo. >> "%SUM%"

call :test MUL_MAT default
call :test MUL_MAT_RELU default
call :test MUL_MAT_HEADSUM default
call :test TOP_K default
call :test TOP_K radix
if not "%TESTRC%"=="0" goto :failed

call :perf TOP_K default
call :perf TOP_K radix

rem paths stay out of parenthesized blocks, a ")" in one would end the block
if not exist "%MODEL%" goto :nomodel
if not exist "%BIN%\llama-bench.exe" goto :nomodel
for %%v in (%ARMS33%) do call :bench %%v
for %%v in (%PROF33%) do call :prof %%v
goto :end

:failed
echo ### a test failed, the perf and model runs are skipped >> "%SUM%"
goto :end

:nomodel
echo ### model or llama-bench not found, the model runs are skipped >> "%SUM%"

:end
call :clearknobs
set "GGML_VK_PERF_LOGGER="
set "LLAMA_INPUT_TIMING="

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

:clearknobs
set "LLAMA_INPUT_THREADS="
set "GGML_VK_DISABLE_MM_HEADSUM="
set "GGML_VK_DISABLE_MM_RELU="
set "GGML_VK_TOPK_RADIX="
goto :eof

rem %1 = the knob of the test and perf steps: default or radix
:knob
call :clearknobs
if /i "%~1"=="radix" set "GGML_VK_TOPK_RADIX=1"
goto :eof

rem %1 = the arm in quotes, "name [VAR=value ...]". = splits call arguments, so the arm comes in one piece and
rem is split here: ANAME = the name, the rest is set one VAR=value at a time
:armenv
call :clearknobs
set "ANAME="
set "AENV="
for /f "tokens=1,*" %%a in (%1) do (
    set "ANAME=%%a"
    set "AENV=%%b"
)
:armenv_next
if not defined AENV goto :eof
for /f "tokens=1,*" %%a in ("!AENV!") do (
    set "%%a"
    set "AENV=%%b"
)
goto :armenv_next

rem after a model run: RETRY=1 when it failed to allocate, or ended with an error. 32 on 03.10 had a crash
rem without a message (exit -1073740791) in the first run after the model load, and the next run with the same
rem -c went through
:retry
set "RETRY=0"
if not "%EC%"=="0" set "RETRY=1"
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" set "RETRY=1"
goto :eof

rem %1 = op, %2 = knob
:test
set "LOG=%LOGS%\33-indexer-%TS%-test-%~1-%~2.log"
call :knob %~2
echo === test %~1, %~2 -^> %LOG%
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o %~1 > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :clearknobs
findstr /C:"tests passed" "%LOG%" >nul
if not "%ERRORLEVEL%"=="0" set "EC=1"
if not "%EC%"=="0" set "RC=%EC%"
if not "%EC%"=="0" set "TESTRC=%EC%"
echo ### test %~1, %~2, exit=%EC% >> "%SUM%"
findstr /C:"tests passed" /C:"FAIL" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem %1 = op, %2 = knob. the k=513 cases only
:perf
set "LOG=%LOGS%\33-indexer-%TS%-perf-%~1-%~2.log"
call :knob %~2
echo === perf %~1, %~2 -^> %LOG%
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o %~1 -p k=513 > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :clearknobs
if not "%EC%"=="0" set "RC=%EC%"
echo ### perf %~1, %~2, exit=%EC% >> "%SUM%"
findstr /C:"us/run" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem %1 = the arm in quotes. the driver frees the memory of the previous process lazily, and a model load
rem right after it can fail to allocate even after SETTLE (README, traps): one more try after another wait
rem (see :retry), the log of the first try stays.
rem the summary takes the host times of the last ubatches of 4096 tokens (the end of the depth fill and the
rem measured ones) and of the last decode step
:bench
set "ARMENV=%~1"
set "TRY=0"
:bench_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
call :armenv "%ARMENV%"
set "LOG=%LOGS%\33-indexer-%TS%-bench-%ANAME%.log"
if %TRY% GTR 1 set "LOG=%LOGS%\33-indexer-%TS%-bench-%ANAME%-try%TRY%.log"
set "LLAMA_INPUT_TIMING=1"
echo === %ARMENV%: llama-bench pp4096 tg32 d%DEPTH33% ctx %CTX33%, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 32 -b 4096 -ub 4096 -c %CTX33% -d %DEPTH33% %LOADMODE% -r 2 %EXTRA% --progress -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "LLAMA_INPUT_TIMING="
call :clearknobs
call :retry
if "%RETRY%"=="1" if %TRY% LSS 2 echo ### %ARMENV%: try %TRY% failed, exit=%EC%, %LOG% >> "%SUM%"
if "%RETRY%"=="1" if %TRY% LSS 2 goto :bench_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### %ARMENV%: llama-bench pp4096 tg32 d%DEPTH33% ctx %CTX33%, try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:"| qwen4exp" /C:"| model" /C:"failed" /C:"pinned memory" /C:"error" "%LOG%" >> "%SUM%"
powershell -NoProfile -Command "$t = Get-Content -LiteralPath '%LOG%'; $t | Select-String -SimpleMatch 'ubatch timing: 4096 tokens' | Select-Object -Last 4 | ForEach-Object { $_.Line }; $t | Select-String -SimpleMatch 'input timing: 4096 tokens' | Select-Object -Last 1 | ForEach-Object { $_.Line }; $t | Select-String -SimpleMatch 'ubatch timing: 1 tokens' | Select-Object -Last 1 | ForEach-Object { $_.Line }" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem %1 = the arm in quotes
:prof
set "ARMENV=%~1"
set "TRY=0"
:prof_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
call :armenv "%ARMENV%"
set "LOG=%LOGS%\33-indexer-%TS%-prof-%ANAME%.log"
if %TRY% GTR 1 set "LOG=%LOGS%\33-indexer-%TS%-prof-%ANAME%-try%TRY%.log"
set "LLAMA_INPUT_TIMING=1"
set "GGML_VK_PERF_LOGGER=1"
echo === %ARMENV%: per-op profile pp4096 d%DEPTH33% ctx %CTX33%, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 0 -b 4096 -ub 4096 -c %CTX33% -d %DEPTH33% %LOADMODE% -r 1 --no-warmup %EXTRA% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_PERF_LOGGER="
set "LLAMA_INPUT_TIMING="
call :clearknobs
call :retry
if "%RETRY%"=="1" if %TRY% LSS 2 echo ### %ARMENV%: try %TRY% failed, exit=%EC%, %LOG% >> "%SUM%"
if "%RETRY%"=="1" if %TRY% LSS 2 goto :prof_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### %ARMENV%: profile pp4096 d%DEPTH33% ctx %CTX33%, try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:"| qwen4exp" /C:"failed" /C:"pinned memory" "%LOG%" >> "%SUM%"
rem only the last graph is measured at the requested depth, the ones before it build the context
powershell -NoProfile -Command "$t = Get-Content -LiteralPath '%LOG%'; $m = $t | Select-String -SimpleMatch 'Vulkan Timings:' | Select-Object -Last 1; if ($m) { $t[($m.LineNumber - 1)..($t.Count - 1)] }" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
