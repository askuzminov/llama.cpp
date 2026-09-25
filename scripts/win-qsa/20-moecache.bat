@echo off
rem what a VRAM cache of the hot MoE experts would give, measured before any cache is written.
rem with -ncmoe the experts of the first layers stay in host memory, and every decode token reads
rem n_expert_used of them per layer over the memory bus. if the same experts come back often, a
rem few slots per layer in VRAM would take most of those reads.
rem LLAMA_MOE_CACHE_STATS makes llama.cpp record the expert ids of every step and replay them
rem through LRU caches of several sizes (src/llama-moecache.cpp). nothing is cached for real and
rem the generated text does not change.
rem
rem one llama-completion run per arm: a prompt of real text, then MOENGEN tokens. the tokens are
rem sampled with a fixed seed, not greedy: a greedy run can fall into a loop, a loop sends the same
rem experts again and again, and the hit rate comes out too high. the arms differ in context and
rem slots, because those decide how much VRAM is left for the cache: the last line of each report
rem is the free VRAM and how many slots fit in it.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem arms, "ctx np": the two server setups in question. a slot gets ctx/np of the context, and the
rem prompt plus MOENGEN has to fit into it
if not defined MOEARMS       set "MOEARMS="262144 1" "131072 2""
rem an LRU of 256 slots that takes one expert per step needs 256 steps to fill, so the run has to
rem be several times longer than that
if not defined MOENGEN       set "MOENGEN=2048"
rem the prompt is the head of MOEPROMPTFILE, 65536 characters is about 16k tokens. the routing
rem depends on the kind of text, so a saved agent session gives numbers closer to the real use
if not defined MOECHARS      set "MOECHARS=65536"
if not defined MOEPROMPTFILE set "MOEPROMPTFILE=%PPLFILE%"
if not defined MOESEED       set "MOESEED=1"
rem decode steps between two reports. the summary keeps the last one, the log keeps all of them,
rem so it shows whether the hit rate has settled
if not defined MOEPERIOD     set "MOEPERIOD=512"
rem -fit off keeps -ncmoe from EXTRA as it is, as the server does. put the -ub of models.ini here
rem too: the compute buffer takes its VRAM from the same pool as the cache
if not defined MOEARGS       set "MOEARGS=-fit off"
if not defined SETTLE        set "SETTLE=30"

if not exist "%BIN%\llama-completion.exe" (
    echo not built: %BIN%\llama-completion.exe
    echo run 00-build.bat first
    exit /b 1
)

if not exist "%MODEL%" (
    echo model not found: %MODEL%
    echo set MODEL in _local.bat
    exit /b 1
)

if /i "%MOEPROMPTFILE%"=="%PPLFILE%" if not exist "%PPLFILE%" call "%~dp0get-wikitext.bat"
if not exist "%MOEPROMPTFILE%" (
    echo prompt file not found: %MOEPROMPTFILE%
    exit /b 1
)

rem a running server holds the model in VRAM: the arm would not load, or the free VRAM in the
rem report would be short by what the server holds
tasklist /fi "imagename eq llama-server.exe" 2>nul | find /i "llama-server.exe" >nul
if not errorlevel 1 (
    echo a llama-server.exe is running, stop it first
    echo it holds VRAM, so the free VRAM in the report would be wrong
    exit /b 1
)

set "PFILE=%LOGS%\20-moecache-%TS%-prompt.txt"
powershell -NoProfile -Command "$t = [IO.File]::ReadAllText('%MOEPROMPTFILE%'); if ($t.Length -gt %MOECHARS%) { $t = $t.Substring(0, %MOECHARS%) }; [IO.File]::WriteAllText('%PFILE%', $t)"
if not exist "%PFILE%" (
    echo could not write %PFILE%
    exit /b 1
)

set "SUM=%LOGS%\20-moecache-%TS%-summary.txt"
echo writing %SUM%
echo ### 20-moecache %TS% > "%SUM%"
echo ### MODEL=%MODEL% >> "%SUM%"
echo ### EXTRA=%EXTRA% LOADMODE=%LOADMODE% MOEARGS=%MOEARGS% >> "%SUM%"
echo ### prompt=%MOEPROMPTFILE% chars=%MOECHARS% n_gen=%MOENGEN% seed=%MOESEED% >> "%SUM%"

set "LLAMA_MOE_CACHE_STATS=%MOEPERIOD%"
set "RC=0"
set "FIRST=1"

for %%v in (%MOEARMS%) do (
    for /f "tokens=1,2" %%a in (%%v) do call :arm %%a %%b
)

set "LLAMA_MOE_CACHE_STATS="
del "%PFILE%" >nul 2>&1

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = context, %2 = slots
:arm
if "%FIRST%"=="1" (
    set "FIRST=0"
) else (
    echo === waiting %SETTLE%s for the gpu to be released
    powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
)

set "LOG=%LOGS%\20-moecache-%TS%-c%~1-np%~2.log"
echo === -c %~1 -np %~2 -^> %LOG%
"%BIN%\llama-completion.exe" -m "%MODEL%" -f "%PFILE%" -c %~1 -np %~2 -fa on %LOADMODE% %EXTRA% %MOEARGS% -no-cnv --no-display-prompt -n %MOENGEN% --ignore-eos --seed %MOESEED% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"

echo. >> "%SUM%"
echo ### -c %~1 -np %~2 exit=%EC% >> "%SUM%"
findstr /c:"eval time =" /c:"failed to" /c:"out of memory" "%LOG%" >> "%SUM%"

rem the last report covers the whole arm. with every expert in VRAM there is no table, only one line
powershell -NoProfile -Command "$t = Get-Content -LiteralPath '%LOG%'; $m = $t | Select-String -SimpleMatch 'MoE expert cache potential' | Select-Object -Last 1; if ($m) { $t[($m.LineNumber - 1)..($t.Count - 1)] | Where-Object { $_ -like '*moe_cache_stats:*' } } else { $t | Where-Object { $_ -like '*moe_cache_stats:*' } | Select-Object -Last 1 }" >> "%SUM%"
goto :eof
