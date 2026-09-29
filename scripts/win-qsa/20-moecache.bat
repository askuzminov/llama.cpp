@echo off
rem what a VRAM cache of the hot MoE experts gives, and where the decode time goes.
rem with -ncmoe the experts of the first layers stay in host memory, and every decode token reads
rem n_expert_used of them per layer over the memory bus. if the same experts come back often, a
rem few slots per layer in VRAM take most of those reads.
rem LLAMA_MOE_CACHE_STATS makes llama.cpp record the expert ids of every step and replay them
rem through simulated caches of several policies, sizes and upload limits (src/llama-moecache.cpp).
rem the simulation does not change the generated text.
rem
rem three kinds of arms, one model load each:
rem   decode (MOEARMS)   llama-completion: a prompt of real text, then MOENGEN tokens sampled with a
rem                      fixed seed. not greedy: a greedy run can fall into a loop, a loop sends the
rem                      same experts again and again, and the hit rate comes out too high
rem   replay (MOEREPLAY) llama-completion with -b = -ub and one generated token: MOEREPLAYCHARS of
rem                      text go through the model ub tokens per step. nothing is sampled, so every
rem                      arm sees the same tokens and the same routing, and the prompt eval time
rem                      compares the arms without the noise of sampling. only the last token of a
rem                      batch is an output and the last layer computes its experts for the outputs
rem                      only, so with -b > -ub the steps without an output would skip that layer
rem   spec (MOESPEC)     llama-speculative-simple with the draft of SPECDRAFT: what draft-mtp and
rem                      ngram-mod give next to the cache. the text is sampled as in the decode arms,
rem                      so it drifts apart between arms; the accept counts show how far. skipped
rem                      when SPECDRAFT is not set
rem set a list to none to skip its arms.
rem
rem an arm can run the real cache (--moe-cache, llama_moe_cache in the same source file). its lines
rem start with "moe_cache: " and go to the summary: the slots it took, the hit rate, the uploads and
rem the host time per step. it uploads at most one expert per layer per step.
rem
rem every arm also runs with GGML_SCHED_PROF: the scheduler prints the host time of its splits every
rem MOEPERIOD graphs, and the summary keeps the last full line of each kind of graph. on CUDA the
rem wait of a CPU split is about the GPU time of the split before it, and "outside" is the host time
rem between two graphs: sampling, the cache update, the draft, and the replay of the statistics once
rem per MOEPERIOD steps. with nvidia-smi on the PATH, nvidia-smi dmon samples the GPU once a second
rem and the summary averages the samples from the first report of the statistics to the end of the
rem generation: SM and memory controller load, power, clock and PCIe traffic.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem decode arms, "ctx np [args]". a slot gets ctx/np of the context, and the prompt plus MOENGEN has
rem to fit into it. -ncmoe in args adds its layers to the -ncmoe of EXTRA, so an arm can only move
rem more layers to the host. the default compares the server setup of EXTRA with the real cache:
rem all 48 MoE layers of the model on the host and the free VRAM in slots. -cmoe would also move
rem the MTP layer, which the cache does not serve. the arms of the first measurement were
rem "262144 1" "131072 2"
if not defined MOEARMS        set "MOEARMS="262144 1" "262144 1 -ncmoe 48 --moe-cache auto""
rem an LRU of 256 slots that takes one expert per step needs 256 steps to fill, so the run has to
rem be several times longer than that
if not defined MOENGEN        set "MOENGEN=2048"
rem the prompt is the head of MOEPROMPTFILE, 65536 characters is about 16k tokens. the routing
rem depends on the kind of text, so a saved agent session gives numbers closer to the real use
if not defined MOECHARS       set "MOECHARS=65536"
if not defined MOEPROMPTFILE  set "MOEPROMPTFILE=%PPLFILE%"
if not defined MOESEED        set "MOESEED=1"
rem decode steps between two reports, and graphs between two sched_prof lines. the summary keeps
rem the last report, the log keeps all of them, so it shows whether the hit rate has settled
if not defined MOEPERIOD      set "MOEPERIOD=512"
rem -fit off keeps -ncmoe from EXTRA as it is, as the server does. put the -ub of models.ini here
rem too: the compute buffer takes its VRAM from the same pool as the cache
if not defined MOEARGS        set "MOEARGS=-fit off"
rem replay arms, "ub [args]": tokens per step. ub 1 is decode, ub 4 is close to a verify step of
rem the MTP draft. --moe-cache auto sizes the cache from the VRAM that is free at this -ub, and a
rem small -ub leaves more of it than the -ub of the server: --moe-cache N pins the size
if not defined MOEREPLAY      set "MOEREPLAY="1" "1 -ncmoe 48 --moe-cache auto" "4 -ncmoe 48 --moe-cache auto""
rem 16384 characters is about 4k tokens, 4k steps at ub 1
if not defined MOEREPLAYCHARS set "MOEREPLAYCHARS=16384"
if not defined MOEREPLAYCTX   set "MOEREPLAYCTX=262144"
rem spec arms, "types [args]": --spec-type and args.
rem -cmoed keeps the experts of the draft in host memory and leaves their VRAM to the cache.
rem ngram-mod drafts 48 to 64 tokens, and a step that long goes over the offload threshold: the
rem experts of the host layers are copied to the GPU and the cache is not used
if not defined MOESPEC        set "MOESPEC="draft-mtp -ncmoe 48 --moe-cache auto" "draft-mtp -ncmoe 48 --moe-cache auto -cmoed" "ngram-mod,draft-mtp -ncmoe 48 --moe-cache auto""
if not defined MOESPECCTX     set "MOESPECCTX=262144"
if not defined SPECNMAX       set "SPECNMAX=6"
if not defined SPECPMIN       set "SPECPMIN=0.6"
rem nvidia-smi dmon during every arm, auto = on cuda when nvidia-smi is on the PATH
if not defined MOEDMON        set "MOEDMON=auto"
if not defined MOEDMONGPU     set "MOEDMONGPU=0"
if not defined SETTLE         set "SETTLE=30"

if /i not "%MOEDMON%"=="auto" goto :dmon_set
set "MOEDMON=0"
if /i not "%BACKEND%"=="cuda" goto :dmon_set
where nvidia-smi >nul 2>&1
if not errorlevel 1 set "MOEDMON=1"
:dmon_set

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

rem the spec arms need the draft and the binary, without them they are skipped
set "SPECSKIP="
if /i "!MOESPEC!"=="none" goto :spec_checked
if not defined SPECDRAFT set "SPECSKIP=SPECDRAFT is not set, see _local.example.bat"
if defined SPECSKIP goto :spec_checked
if not exist "%SPECDRAFT%" set "SPECSKIP=draft not found: %SPECDRAFT%"
if not exist "%BIN%\llama-speculative-simple.exe" set "SPECSKIP=not built: %BIN%\llama-speculative-simple.exe"
:spec_checked

rem the prompt of the decode and spec arms, and the text of the replay arms
set "PFILE=%LOGS%\20-moecache-%TS%-prompt.txt"
set "RFILE=%LOGS%\20-moecache-%TS%-replay.txt"
powershell -NoProfile -Command "$t = [IO.File]::ReadAllText($env:MOEPROMPTFILE); $p = $t; $r = $t; if ($p.Length -gt [int]$env:MOECHARS) { $p = $p.Substring(0, [int]$env:MOECHARS) }; if ($r.Length -gt [int]$env:MOEREPLAYCHARS) { $r = $r.Substring(0, [int]$env:MOEREPLAYCHARS) }; [IO.File]::WriteAllText($env:PFILE, $p); [IO.File]::WriteAllText($env:RFILE, $r)"
if not exist "%PFILE%" (
    echo could not write %PFILE%
    exit /b 1
)
if not exist "%RFILE%" (
    echo could not write %RFILE%
    exit /b 1
)

set "SUM=%LOGS%\20-moecache-%TS%-summary.txt"
echo writing %SUM%
echo ### 20-moecache %TS% > "%SUM%"
echo ### MODEL=%MODEL% >> "%SUM%"
echo ### EXTRA=%EXTRA% LOADMODE=%LOADMODE% MOEARGS=%MOEARGS% >> "%SUM%"
echo ### prompt=%MOEPROMPTFILE% chars=%MOECHARS% n_gen=%MOENGEN% seed=%MOESEED% period=%MOEPERIOD% >> "%SUM%"
echo ### replay chars=%MOEREPLAYCHARS% ctx=%MOEREPLAYCTX%; spec draft=%SPECDRAFT% ctx=%MOESPECCTX% n_max=%SPECNMAX% p_min=%SPECPMIN%; dmon=%MOEDMON% gpu=%MOEDMONGPU% >> "%SUM%"

set "RC=0"

rem the cache matmuls give every expert that is not cached the zero slot, so a row of ids repeats a
rem slot. the backend test checks that the device computes every use of a slot, no model needed
if exist "%BIN%\test-backend-ops.exe" (
    set "TLOG=%LOGS%\20-moecache-%TS%-test.log"
    echo === test-backend-ops MUL_MAT_ID_REPEAT -^> !TLOG!
    "%BIN%\test-backend-ops.exe" test -o MUL_MAT_ID_REPEAT > "!TLOG!" 2>&1
    set "EC=!ERRORLEVEL!"
    if not "!EC!"=="0" set "RC=!EC!"
    echo. >> "%SUM%"
    echo ### test-backend-ops MUL_MAT_ID_REPEAT exit=!EC! >> "%SUM%"
    findstr /c:"FAIL" /c:"tests passed" /c:"CUDA error" "!TLOG!" >> "%SUM%"
)

set "LLAMA_MOE_CACHE_STATS=%MOEPERIOD%"
set "GGML_SCHED_PROF=%MOEPERIOD%"
set "FIRST=1"
set "ARM=0"

if /i "!MOEARMS!"=="none" goto :decode_done
for %%v in (%MOEARMS%) do (
    for /f "tokens=1,2,*" %%a in (%%v) do call :arm %%a %%b "%%c"
)
:decode_done

if /i "!MOEREPLAY!"=="none" goto :replay_done
for %%v in (%MOEREPLAY%) do (
    for /f "tokens=1,*" %%a in (%%v) do call :arm_replay %%a "%%b"
)
:replay_done

if /i "!MOESPEC!"=="none" goto :spec_done
if not defined SPECSKIP goto :spec_run
echo. >> "%SUM%"
echo ### spec arms skipped: %SPECSKIP% >> "%SUM%"
goto :spec_done
:spec_run
set "SPECFIRST=1"
rem the types are passed in quotes: call splits its arguments at commas
for %%v in (%MOESPEC%) do (
    for /f "tokens=1,*" %%a in (%%v) do call :arm_spec "%%a" "%%b"
)
:spec_done

set "LLAMA_MOE_CACHE_STATS="
set "GGML_SCHED_PROF="
del "%PFILE%" >nul 2>&1
del "%RFILE%" >nul 2>&1

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = context, %2 = slots, %3 = args of the arm in quotes
:arm
call :next_arm
set "BASE=%LOGS%\20-moecache-%TS%-arm%ARM%-dec-c%~1-np%~2"
set "LOG=%BASE%.log"
echo === arm %ARM%: decode -c %~1 -np %~2 %~3 -^> %LOG%
call :dmon_start
"%BIN%\llama-completion.exe" -m "%MODEL%" -f "%PFILE%" -c %~1 -np %~2 -fa on %LOADMODE% %EXTRA% %MOEARGS% %~3 -no-cnv --no-display-prompt -n %MOENGEN% --ignore-eos --seed %MOESEED% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :dmon_stop
if not "%EC%"=="0" set "RC=%EC%"

echo. >> "%SUM%"
echo ### arm %ARM%: decode -c %~1 -np %~2 %~3 exit=%EC% >> "%SUM%"
findstr /c:"eval time =" /c:"failed to" /c:"out of memory" /c:"moe_cache: " "%LOG%" >> "%SUM%"
set "SUMMODE=full"
call :sum_all
goto :eof

rem %1 = tokens per step, %2 = args of the arm in quotes
:arm_replay
call :next_arm
set "BASE=%LOGS%\20-moecache-%TS%-arm%ARM%-rep-ub%~1"
set "LOG=%BASE%.log"
echo === arm %ARM%: replay -b %~1 -ub %~1 %~2 -^> %LOG%
call :dmon_start
"%BIN%\llama-completion.exe" -m "%MODEL%" -f "%RFILE%" -c %MOEREPLAYCTX% -fa on %LOADMODE% %EXTRA% %MOEARGS% -b %~1 -ub %~1 %~2 -no-cnv --no-display-prompt -n 1 --seed %MOESEED% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :dmon_stop
if not "%EC%"=="0" set "RC=%EC%"

echo. >> "%SUM%"
echo ### arm %ARM%: replay -b %~1 -ub %~1 %~2 exit=%EC% >> "%SUM%"
findstr /c:"prompt eval time =" /c:"failed to" /c:"out of memory" /c:"moe_cache: " "%LOG%" >> "%SUM%"
set "SUMMODE=full"
call :sum_all
goto :eof

rem %1 = spec types in quotes, %2 = args of the arm in quotes
:arm_spec
call :next_arm
set "STAG=%~1"
if not defined STAG goto :eof
set "STAG=%STAG:,=+%"
set "BASE=%LOGS%\20-moecache-%TS%-arm%ARM%-spec-%STAG%"
set "LOG=%BASE%.log"
echo === arm %ARM%: spec %~1 %~2 -^> %LOG%
call :dmon_start
rem -b holds the whole prompt, llama-speculative-simple decodes it in one batch
"%BIN%\llama-speculative-simple.exe" -m "%MODEL%" -md "%SPECDRAFT%" --spec-type %~1 --spec-draft-n-min 0 --spec-draft-n-max %SPECNMAX% --spec-draft-p-min %SPECPMIN% -f "%PFILE%" -c %MOESPECCTX% -b %MOECHARS% -fa on %LOADMODE% %EXTRA% %MOEARGS% %~2 -n %MOENGEN% --ignore-eos --seed %MOESEED% -lv 4 > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
call :dmon_stop
if not "%EC%"=="0" set "RC=%EC%"

echo. >> "%SUM%"
echo ### arm %ARM%: spec %~1 %~2 exit=%EC% >> "%SUM%"
findstr /c:"I encoded " /c:"I decoded " /c:"I n_drafted" /c:"I n_accept" /c:"I accept " /c:"common_specu: statistics" /c:"eval time =" /c:"failed to" /c:"out of memory" /c:"moe_cache: " "%LOG%" >> "%SUM%"
rem the arms replay different text, so only the first one gets the full tables
set "SUMMODE=short"
if "%SPECFIRST%"=="1" set "SUMMODE=full"
set "SPECFIRST=0"
call :sum_all
goto :eof

:next_arm
set /a ARM+=1
if "%FIRST%"=="1" (
    set "FIRST=0"
) else (
    echo === waiting %SETTLE%s for the gpu to be released
    powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
)
goto :eof

rem nvidia-smi buffers its output in a pipe until it exits, so dmon runs in chunks of 10 samples
rem while the run file exists. T0MS is the time of day right before the model starts, the log
rem timestamps count from the start of the process
:dmon_start
if not "%MOEDMON%"=="1" goto :eof
set "DLOG=%BASE%-dmon.log"
set "DRUN=%DLOG%.run"
set "DDONE=%DLOG%.done"
del "%DDONE%" >nul 2>&1
echo.> "%DRUN%"
start "" /b powershell -NoProfile -Command "$stop = (Get-Date).AddHours(4); while ((Test-Path -LiteralPath $env:DRUN) -and ((Get-Date) -lt $stop)) { $s = Get-Date; nvidia-smi dmon -i $env:MOEDMONGPU -s puct -o T -c 10 2>&1 | Add-Content -LiteralPath $env:DLOG; if (((Get-Date) - $s).TotalSeconds -lt 5) { Start-Sleep -Seconds 5 } }; Set-Content -LiteralPath $env:DDONE -Value done"
for /f %%t in ('powershell -NoProfile -Command "[int][math]::Floor((Get-Date).TimeOfDay.TotalMilliseconds)"') do set "T0MS=%%t"
goto :eof

:dmon_stop
if not "%MOEDMON%"=="1" goto :eof
del "%DRUN%" >nul 2>&1
powershell -NoProfile -Command "$n = 0; while ((-not (Test-Path -LiteralPath $env:DDONE)) -and ($n -lt 30)) { Start-Sleep -Milliseconds 500; $n++ }"
del "%DDONE%" >nul 2>&1
goto :eof

:sum_all
call :sum_stats
call :sum_prof
call :sum_dmon
goto :eof

rem the last report covers the whole arm. with every expert in VRAM there is no table, only one
rem line. short keeps the counts, the header and the rows of the real cache (*)
:sum_stats
powershell -NoProfile -Command "$t = Get-Content -LiteralPath $env:LOG; $m = $t | Select-String -SimpleMatch 'MoE expert cache potential' | Select-Object -Last 1; if ($m) { $r = $t[($m.LineNumber - 1)..($t.Count - 1)] | Where-Object { $_ -like '*moe_cache_stats:*' }; if ($env:SUMMODE -eq 'short') { $r = $r | Where-Object { $_ -match 'decode steps|host expert reads|replayed|fits|slots \| VRAM total \| ins|\d\* \|' } } } else { $r = $t | Where-Object { $_ -like '*moe_cache_stats:*' } | Select-Object -Last 1 }; foreach ($x in $r) { $x.Substring($x.IndexOf('moe_cache_stats:')) }" >> "%SUM%"
goto :eof

rem one line per scheduler and split count: the last one of the most graphs, a full period when
rem there was one
:sum_prof
powershell -NoProfile -Command "$best = @{}; $keys = New-Object System.Collections.ArrayList; foreach ($l in (Get-Content -LiteralPath $env:LOG)) { $i = $l.IndexOf('sched_prof '); if ($i -lt 0) { continue }; $s = $l.Substring($i); if ($s -match '^sched_prof (\d+): (\d+) graphs of (\S+) splits') { $k = $matches[1] + ' ' + $matches[3]; $n = [int64]$matches[2]; if (-not $best.ContainsKey($k)) { [void]$keys.Add($k); $best[$k] = @($n, $s) } elseif ($n -ge $best[$k][0]) { $best[$k] = @($n, $s) } } }; foreach ($k in $keys) { $best[$k][1] }" >> "%SUM%"
goto :eof

rem the window starts at the first report of the statistics, else at the start of the cache, and
rem ends at the perf print of llama-completion or the encoded line of llama-speculative-simple
:sum_dmon
if not "%MOEDMON%"=="1" goto :eof
powershell -NoProfile -Command "$lines = Get-Content -LiteralPath $env:LOG; $ft = { param($p) foreach ($l in $lines) { $m = [regex]::Match($l, $p); if ($m.Success) { return (60 * [double]$m.Groups[1].Value + [double]$m.Groups[2].Value + [double]$m.Groups[3].Value / 1000 + [double]$m.Groups[4].Value / 1000000) } }; return -1.0 }; $ts = '(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) '; $tE = & $ft ('^' + $ts + 'I (common_perf_print|encoded |decoded )'); if ($tE -lt 0) { $tE = [double]::PositiveInfinity }; $tS = & $ft ('^' + $ts + 'W moe_cache_stats: experts '); $how = 'first report to end'; if (($tS -lt 0) -or ($tS -gt $tE)) { $tS = & $ft ($ts + 'W moe_cache: [^,]+, \d+ of \d+ MoE layers'); $how = 'cache start to end' }; if (($tS -lt 0) -or ($tS -gt $tE)) { $tS = 0.0; $how = 'whole run' }; $t0 = [double]$env:T0MS / 1000; $inv = [Globalization.CultureInfo]::InvariantCulture; $want = @('sm', 'mem', 'pwr', 'pclk', 'rxpci', 'txpci'); $unit = @{ sm = '%%'; mem = '%%'; pwr = ' W'; pclk = ' MHz'; rxpci = ' MB/s'; txpci = ' MB/s' }; $sum = @{}; $cnt = @{}; $cols = $null; $n = 0; $t1 = 0.0; $t2 = 0.0; $dl = @(); if (Test-Path -LiteralPath $env:DLOG) { $dl = Get-Content -LiteralPath $env:DLOG }; foreach ($l in $dl) { if ($l -match '^#\s*Time') { $cols = ($l -replace '^#\s*', '').Trim() -split '\s+'; continue }; if (($null -eq $cols) -or $l.StartsWith('#')) { continue }; $f = $l.Trim() -split '\s+'; if (($f.Count -ne $cols.Count) -or ($f[0] -notmatch '^(\d+):(\d+):(\d+)$')) { continue }; $t = 3600 * [double]$matches[1] + 60 * [double]$matches[2] + [double]$matches[3] - $t0; if ($t -lt -43200) { $t += 86400 }; if (($t -lt $tS) -or ($t -gt $tE)) { continue }; if ($n -eq 0) { $t1 = $t }; $t2 = $t; $n++; for ($i = 1; $i -lt $cols.Count; $i++) { $c = $cols[$i]; $v = 0.0; if (($want -contains $c) -and [double]::TryParse($f[$i], [Globalization.NumberStyles]::Float, $inv, [ref]$v)) { $sum[$c] = [double]$sum[$c] + $v; $cnt[$c] = [int]$cnt[$c] + 1 } } }; if ($n -eq 0) { 'dmon: no samples in the window, see ' + $env:DLOG } else { $p = foreach ($c in $want) { if ([int]$cnt[$c] -gt 0) { $c + ' ' + [math]::Round($sum[$c] / $cnt[$c]) + $unit[$c] } }; 'dmon: ' + $n + ' samples from ' + [math]::Round($t1) + ' s to ' + [math]::Round($t2) + ' s after the start (' + $how + '): ' + ($p -join ', ') }" >> "%SUM%"
goto :eof
