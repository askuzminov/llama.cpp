@echo off
rem how the generation speed falls with the depth of the context, and which op pays for it.
rem 03 and 04 sweep the prefill, 07 times the ops of one prefill graph; none of them says
rem what a decode pass costs at depth. the question is not academic: QSA caps the attention
rem itself at a fixed budget of blocks, but the block keys of the indexer are rebuilt from
rem the whole cache on every pass and on every full attention layer
rem (src/models/qwen4exp.cpp, build_qsa_top_k: get_rows -> 4 cont -> add -> scale -> norm
rem -> rope). that work is O(n_kv) and on the prefill it is spread over n_ubatch tokens, on
rem the decode it is paid per token. so the decode is expected to grow linearly with depth.
rem
rem phase 1 gives the curve, phase 2 gives the per-op breakdown at two points of it, so the
rem growth can be attributed. cuda ignores GGML_VK_PERF_LOGGER, phase 2 is vulkan only.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem глубины кривой. 0 это контекст короче бюджета индексатора, там весь индексатор
rem пропускается (bypass при n_kv <= 2052), поэтому первая точка после нуля обязана быть
rem выше порога, иначе ступенька спишется на глубину. верхняя точка плюс DECNGEN не должна
rem вылезти за 262144 = n_ctx_train, llama-bench берёт n_ctx = n_prompt + n_gen + n_depth
if not defined DECDEPTHS set "DECDEPTHS=0,4096,16384,32768,65536,131072,262080"
rem сколько токенов генерировать в замере. 64 хватает: разброс по глубине больше разброса
rem между повторами, а на 262080 каждый лишний токен стоит заметно
if not defined DECNGEN set "DECNGEN=64"
rem батч только для набора глубины, на сам замер не влияет: на декоде n_tokens = 1
if not defined DECB set "DECB=4096"
if not defined DECUB set "DECUB=2048"
rem точки, в которых снимается разбор по операциям. две: мелкая и глубокая, разница между
rem ними и показывает, какие узлы растут с n_kv, а какие стоят на месте
if not defined DECPERFDEPTHS set "DECPERFDEPTHS=16384 131072"
if not defined SETTLE set "SETTLE=30"

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

set "SUM=%LOGS%\19-decode-depth-%TS%-summary.txt"
echo writing %SUM%
echo ### 19-decode-depth %TS% > "%SUM%"
echo ### MODEL=%MODEL% >> "%SUM%"
echo ### depths=%DECDEPTHS% n_gen=%DECNGEN% reps=%REPS% >> "%SUM%"

set "RC=0"

rem ---------------------------------------------------------------
rem  фаза 1: кривая tg по глубине, одна загрузка модели на все точки
rem ---------------------------------------------------------------
set "LOG=%LOGS%\19-decode-depth-%TS%-tg.log"
echo === tg %DECNGEN% at depths %DECDEPTHS% -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 0 -n %DECNGEN% -b %DECB% -ub %DECUB% -d %DECDEPTHS% %LOADMODE% -r %REPS% %EXTRA% --progress -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo. >> "%SUM%"
echo ### tg curve exit=%EC% >> "%SUM%"
type "%LOG%" >> "%SUM%"

rem ---------------------------------------------------------------
rem  фаза 2: разбор по операциям, одна загрузка модели на глубину
rem ---------------------------------------------------------------
rem фаза 1 идёт на любом бэкенде, фаза 2 это только таймингы vulkan
if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, skipping phase 2
    goto :phase2_done
)

set "GGML_VK_PERF_LOGGER=1"
for %%d in (%DECPERFDEPTHS%) do call :perf %%d
set "GGML_VK_PERF_LOGGER="
:phase2_done

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = глубина, на которой снимается граф декода
:perf
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
set "LOG=%LOGS%\19-decode-depth-%TS%-perf-d%~1.log"
echo === per-op at d%~1 -^> %LOG%
rem -n 1: набор глубины даёт свои графы, но последний Vulkan Timings это уже граф декода
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p 0 -n 1 -b %DECB% -ub %DECUB% -d %~1 %LOADMODE% -r 1 --no-warmup %EXTRA% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"

echo. >> "%SUM%"
echo ### per-op decode graph at d%~1 exit=%EC% >> "%SUM%"
findstr /C:"| qwen4exp" /C:"failed to load model" /C:"allocation of size" "%LOG%" >> "%SUM%"

rem только последний граф это декод, все предыдущие набирали глубину
powershell -NoProfile -Command "$t = Get-Content -LiteralPath '%LOG%'; $m = $t | Select-String -SimpleMatch 'Vulkan Timings:' | Select-Object -Last 1; if ($m) { $t[($m.LineNumber - 1)..($t.Count - 1)] }" >> "%SUM%"
goto :eof
