@echo off
rem prefill against the depth of the context (02.10). a server run of a ~200K prompt on 5bdd6c0c9 (-c 262144,
rem -ub 4096) went 630 t/s per 4096-token chunk at ~10K, ~470 at ~54K, 316 at ~125K, and from n_kv 131072
rem about 200, with a "failed to allocate pinned memory" warning per chunk. one cause behind both, the slope
rem and the step: the qwen4exp indexer scored all 4 heads in one f32 tensor of n_ctx x ubatch x 4 bytes in the
rem graph the server reserves its buffers with (n_kv = n_ctx), 4 GiB at 262144 x 4096, past the 2 GiB buffer
rem limit of the AMD driver, so the reserved graph had those ops on the CPU. the graphs of the prompt chunks
rem have them on the GPU, the scheduler then re-reserved on every chunk (a full GPU sync, a new plan, the
rem compute buffer grown again), and from n_kv 131072 the chunks put them on the CPU as well. n_ctx x ubatch
rem up to 512M (131072 x 4096, 262144 x 2048) stays at 2 GiB and never triggered it. the indexer now scores one
rem head at a time (n_kv x ubatch bytes per tensor, 1 GiB at 262144 x 4096). llama-bench sizes its context to
rem the test (depth + prompt), so its reserved graph is the measured one: -c (added 02.10) reserves for a
rem larger context, as the server does.
rem   1 llama-bench pp4096 over DEPTH32 for each arm of ARMS32, "build ubatch [ctx]": new = this build (BIN),
rem     old = the llama-bench.exe in OLDBIN32 (set in _local.bat, empty skips the arm), ctx = -c (empty = the
rem     context of the test; an older llama-bench has no -c). each depth is filled first, outside the timing,
rem     so an arm costs about the sum of the depths in prefill time
rem   2 vulkan only: per-op profile of one ubatch at the arms of PROF32, "build ubatch depth [ctx]"
rem     (GGML_VK_PERF_LOGGER, the last graph of the log is the measured one). the wall time of that ubatch is
rem     its size / t/s of step 1; the gap to "Total time" of the profile is host work (mask build, input
rem     upload, graph build)
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

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

set "SUM=%LOGS%\32-depth-%TS%-summary.txt"
set "RC=0"
set "GGML_VK_PERF_LOGGER="
echo writing %SUM%
echo ### 32-depth %TS%, depths %DEPTH32%, arms %ARMS32%, profiles %PROF32% > "%SUM%"
echo ### new = %BIN% >> "%SUM%"
echo ### old = %OLDBIN32% >> "%SUM%"
echo. >> "%SUM%"

for %%v in (%ARMS32%) do call :sweep %%~v
if /i not "%BACKEND%"=="vulkan" goto :end
for %%v in (%PROF32%) do call :prof %%~v

:end
set "GGML_VK_PERF_LOGGER="

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = new or old, sets ABIN. an old arm without OLDBIN32 is skipped
:pickbin
set "ABIN="
if /i "%~1"=="new" set "ABIN=%BIN%"
if /i not "%~1"=="old" goto :eof
if "%OLDBIN32%"=="" goto :eof
if exist "%OLDBIN32%\llama-bench.exe" set "ABIN=%OLDBIN32%"
goto :eof

rem %1 = build, %2 = ubatch, %3 = ctx or empty. the driver frees the memory of the previous process lazily,
rem and a model load right after it can fail to allocate even after SETTLE (README, traps): one more try
rem after another wait
:sweep
call :pickbin %~1
if "%ABIN%"=="" goto :nobin
set "CARG="
set "CTAG=test"
if not "%~3"=="" set "CARG=-c %~3"
if not "%~3"=="" set "CTAG=%~3"
set "TRY=0"
:sweep_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\32-depth-%TS%-sweep-%~1-ub%~2-c%CTAG%.log"
echo === %~1 build, llama-bench pp4096 ub%~2 ctx %CTAG% at d %DEPTH32%, try %TRY% -^> %LOG%
"%ABIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 4096 -n 0 -b 4096 -ub %~2 %CARG% -d %DEPTH32% %LOADMODE% -r 1 %EXTRA% --progress -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :sweep_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### %~1 build, llama-bench pp4096 ub%~2 ctx %CTAG%, try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:"| qwen4exp" /C:"| model" /C:"build:" /C:"failed" /C:"pinned memory" /C:"error" /C:"invalid" "%LOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof

:nobin
echo ### %~1 build: no llama-bench.exe (set OLDBIN32 in _local.bat for the old build), skipped >> "%SUM%"
echo. >> "%SUM%"
goto :eof

rem %1 = build, %2 = ubatch, %3 = depth, %4 = ctx or empty
:prof
call :pickbin %~1
if "%ABIN%"=="" goto :nobin
set "CARG="
set "CTAG=test"
if not "%~4"=="" set "CARG=-c %~4"
if not "%~4"=="" set "CTAG=%~4"
set "TRY=0"
:prof_try
set /a TRY+=1
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\32-depth-%TS%-prof-%~1-ub%~2-d%~3-c%CTAG%.log"
set "GGML_VK_PERF_LOGGER=1"
echo === %~1 build, per-op profile pp%~2 ub%~2 d%~3 ctx %CTAG%, try %TRY% -^> %LOG%
"%ABIN%\llama-bench.exe" -m "%MODEL%" -fa on -p %~2 -n 0 -b %~2 -ub %~2 %CARG% -d %~3 %LOADMODE% -r 1 --no-warmup %EXTRA% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_PERF_LOGGER="
findstr /C:"failed to allocate" /C:"Device memory allocation" "%LOG%" >nul
if "%ERRORLEVEL%"=="0" if %TRY% LSS 2 goto :prof_try
if not "%EC%"=="0" set "RC=%EC%"
echo ### %~1 build, profile pp%~2 ub%~2 d%~3 ctx %CTAG%, try %TRY%, exit=%EC% >> "%SUM%"
findstr /C:"| qwen4exp" /C:"failed" /C:"pinned memory" /C:"invalid" "%LOG%" >> "%SUM%"
rem only the last graph is measured at the requested depth, the ones before it build the context
powershell -NoProfile -Command "$t = Get-Content -LiteralPath '%LOG%'; $m = $t | Select-String -SimpleMatch 'Vulkan Timings:' | Select-Object -Last 1; if ($m) { $t[($m.LineNumber - 1)..($t.Count - 1)] }" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
