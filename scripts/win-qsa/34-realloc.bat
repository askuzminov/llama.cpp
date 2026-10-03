@echo off
rem why the scheduler reserves its buffers again on every ubatch (03.10). in the 33 run at -c 262144 -ub 4096 the graph
rem phase of each ubatch of the depth fill waited for the gpu (4.9 to 5.8 s, the gpu time of the ubatch before) and
rem took 555 ms on the measured ubatch: a sync of all backends, a new buffer plan and a larger buffer each time. on the
rem Mac the same chain starts when the graph the buffers are reserved with (n_kv = n_ctx) has an op that supports_op
rem sends to the CPU because a tensor is over the buffer limit, while the real graphs run it on the gpu: the first
rem real graph reserves again with its own (small) sizes, and each later graph with a larger n_kv no longer fits. the
rem KQ mask at n_ctx 262144 and a 4096-token ubatch is exactly 2 GiB (f16): on the Mac with the limit set to 2 GiB - 1
rem its FILL and SET_ROWS go to the CPU in the reserved graph and the chain follows, with the limit at 2 GiB it does not.
rem GGML_SCHED_LOG_REALLOC=1 prints the limits of the device, the splits of every reserved graph (with the ops that
rem left the gpu) and the cause of each re-reserve.
rem   llama-bench pp4096 at d8192 (two ubatches of fill, then the measured one) for each -c of CTX34. 261888 is the
rem   largest multiple of 256 with the mask under 2 GiB, 131072 has it at 1 GiB
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
set "GGML_SCHED_LOG_REALLOC="
set "LLAMA_INPUT_TIMING="
echo writing %SUM%
echo ### 34-realloc %TS%, contexts %CTX34% > "%SUM%"
echo. >> "%SUM%"

for %%c in (%CTX34%) do call :bench %%c

set "GGML_SCHED_LOG_REALLOC="
set "LLAMA_INPUT_TIMING="

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = the context. the driver frees the memory of the previous process lazily, and a model load right after it can
rem fail even after SETTLE (README, traps): one more try after another wait, the log of the first try stays
:bench
set "TRY=0"
:bench_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\34-realloc-%TS%-c%~1.log"
if %TRY% GTR 1 set "LOG=%LOGS%\34-realloc-%TS%-c%~1-try%TRY%.log"
set "GGML_SCHED_LOG_REALLOC=1"
set "LLAMA_INPUT_TIMING=1"
echo === -c %~1: llama-bench pp4096 d8192, try %TRY% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 0 -b 4096 -ub 4096 -c %~1 -d 8192 %LOADMODE% -r 1 %EXTRA% --progress -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_SCHED_LOG_REALLOC="
set "LLAMA_INPUT_TIMING="
set "RETRY=0"
if not "%EC%"=="0" set "RETRY=1"
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" set "RETRY=1"
if "%RETRY%"=="1" if %TRY% LSS 2 echo ### -c %~1: try %TRY% failed, exit=%EC%, %LOG% >> "%SUM%"
if "%RETRY%"=="1" if %TRY% LSS 2 goto :bench_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### -c %~1: llama-bench pp4096 d8192, try %TRY%, exit=%EC% >> "%SUM%"
rem the limits, the splits of the reserved graphs, the causes of the re-reserves, the host time of each ubatch, the result
findstr /B /C:"ggml_vulkan: max" /C:"sched " /C:"  split" /C:"    " /C:"galloc " /C:"ubatch timing" /C:"llama-bench: benchmark" /C:"ggml_uncaught" "%LOG%" >> "%SUM%"
findstr /C:"| qwen4exp" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
