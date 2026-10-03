@echo off
rem the scheduler planned its buffers again on every ubatch of a llama-bench prompt (03.10). the run of 03.10 11:32
rem found the cause: llama-bench leaves n_outputs_max at n_batch, so the graph the buffers are reserved with has the
rem logits of all 4096 tokens, a [248320, 4096] f32 matmul of 4 GB over the 2 GiB buffer limit of the driver, run on the
rem CPU there (3 splits, 7157 nodes). a prompt ubatch needs the logits of one token at most, on the gpu (2 splits, 7153
rem nodes): the first ubatch plans again for its own small sizes, and each next one with a larger KV again. the graph
rem phase of every ubatch then waits for the gpu (a sync, 4.9 to 5.8 s in the depth fill) and costs 125 ms at d8192,
rem 555 ms at d122880. llama-server reserves with few outputs and never had it (checked on the Mac: no new plan with or
rem without the fix). fix: the reserve at the start plans the prompt graph with one output per sequence last, and a
rem plan made again for a prompt ubatch while computing is made at the full KV. the run of 03.10 11:58 also showed
rem the cause of the silent crashes at the first ubatch: vk::Queue::submit: ErrorOutOfDeviceMemory (memory is at the
rem edge at -c 262144 with this model; the first version of the fix planned at the full KV in the middle of the run
rem and crashed so twice). LLAMA_REPLAN_DISABLE=1 turns the fix off. GGML_SCHED_LOG_REALLOC=1 prints the limits of the
rem device, the splits of every planned graph and the cause of each new plan.
rem   llama-bench pp4096 at DEPTH34 with -c CTX34 for each arm of ARMS34, "name [VAR=value ...]"; the depth is filled
rem   once per arm, outside the timing
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 34-realloc is vulkan only
    exit /b 0
)

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

set "SUM=%LOGS%\34-realloc-%TS%-summary.txt"
set "RC=0"
call :clearknobs
echo writing %SUM%
echo ### 34-realloc %TS%, depth %DEPTH34%, ctx %CTX34%, arms %ARMS34% > "%SUM%"
echo. >> "%SUM%"

for %%v in (%ARMS34%) do call :bench %%v

call :clearknobs

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

:clearknobs
set "LLAMA_REPLAN_DISABLE="
set "GGML_SCHED_LOG_REALLOC="
set "LLAMA_INPUT_TIMING="
goto :eof

rem %1 = the arm in quotes, "name [VAR=value ...]". = splits call arguments, so the arm comes in one piece and is split
rem here: ANAME = the name, the rest is set one VAR=value at a time
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

rem %1 = the arm in quotes. the driver frees the memory of the previous process lazily, and a model load right after it
rem can fail even after SETTLE (README, traps): one more try after another wait, the log of the first try stays
:bench
set "ARMENV=%~1"
set "TRY=0"
:bench_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
call :armenv "%ARMENV%"
set "LOG=%LOGS%\34-realloc-%TS%-%ANAME%.log"
if %TRY% GTR 1 set "LOG=%LOGS%\34-realloc-%TS%-%ANAME%-try%TRY%.log"
set "GGML_SCHED_LOG_REALLOC=1"
set "LLAMA_INPUT_TIMING=1"
echo === %ARMENV%: llama-bench pp4096 d%DEPTH34% ctx %CTX34%, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 0 -b 4096 -ub 4096 -c %CTX34% -d %DEPTH34% %LOADMODE% -r 2 %EXTRA% --progress -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :clearknobs
set "RETRY=0"
if not "%EC%"=="0" set "RETRY=1"
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" set "RETRY=1"
if "%RETRY%"=="1" if %TRY% LSS 2 echo ### %ARMENV%: try %TRY% failed, exit=%EC%, %LOG% >> "%SUM%"
if "%RETRY%"=="1" if %TRY% LSS 2 goto :bench_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### %ARMENV%: llama-bench pp4096 d%DEPTH34% ctx %CTX34%, try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:"| qwen4exp" /C:"ggml_uncaught" "%LOG%" >> "%SUM%"
rem the plans: counts, then the causes; the host time of the first ubatches of 4096 tokens and of the last ones (the end
rem of the depth fill and the measured ones)
powershell -NoProfile -Command "$t = Get-Content -LiteralPath '%LOG%'; 'plans at start (reserve): ' + @($t | Select-String -SimpleMatch -Pattern 'sched reserve:').Count + ', plans while computing (re-reserve): ' + @($t | Select-String -SimpleMatch -Pattern 'sched re-reserve:').Count; $t | Select-String -Pattern '^(sched realloc|galloc realloc)' | Select-Object -First 6 | ForEach-Object { $_.Line }; $u = @($t | Select-String -SimpleMatch -Pattern 'ubatch timing: 4096 tokens'); $u | Select-Object -First 3 | ForEach-Object { $_.Line }; '...'; $u | Select-Object -Last 4 | ForEach-Object { $_.Line }" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
