@echo off
rem the memory of llama-server with -np 1, where the prefill and the decode take turns. one
rem llama-server start per arm, -b = -ub. two kinds of arms:
rem   UBMEMSTATIC  the static layout. the compute buffer is reserved at start for the worst ubatch
rem                at the full context and stays. --moe-cache auto sizes itself at the first decode
rem                from the free VRAM and keeps that size
rem   UBMEMARMS    --phase-mem. a prompt batch releases the cache and gets the largest ubatch up to
rem                -ub that fits the free memory at the real KV depth, planned again when the depth
rem                grows. a decode batch gets a small compute buffer, the cache takes the rest and
rem                uploads the experts that were hot before the prompt. so -ub can be larger than
rem                the static layout holds at the full context
rem the cache and --phase-mem leave 1 GiB free. only the backend pools grow later, when an op needs
rem them: the CUDA pool, the vulkan staging and dequant buffers. a pool that grows at the long
rem prefill has only that margin, and after it the cache gets fewer slots. the phases of an arm, in
rem the worst order:
rem   load     start, reserve, warmup, until /health answers
rem   short1   a short prompt and UBMEMNGEN tokens: the first decode sizes the cache
rem   long     UBMEMFILL full ubatches and a half one of the largest ub of all arms, the same
rem            prompt for every arm, and 16 tokens
rem   short2   short1 again: the decode after the pools have grown
rem an arm survives when every request answers 200 with timings and /health still answers after
rem the last one. a sampler writes the GPU memory of the server process and of its adapter about
rem once a second (the WMI GPU counters, the same on NVIDIA and AMD) and on cuda nvidia-smi too.
rem the summary keeps the maximum per phase. when VRAM is full, the NVIDIA windows driver (sysmem
rem fallback) and vulkan on a UMA device put an allocation in shared host memory instead of
rem failing: then shared grows after short1 and short2 decodes slower, the summary flags both.
rem the table: slots is the cache size at the first and at the last allocation, pp_ub the smallest
rem and the largest prompt ubatch, sw_n the phase switches, sw_s the seconds in the switches and in
rem the cache allocations with their uploads.
rem vulkan: an op with a tensor above the device buffer limit, or above maxStorageBufferRange
rem without 64-bit indexing, goes to the cpu without a message. the table flags a static arm with
rem more splits than the first static arm. --phase-mem does not take a ubatch with more splits than
rem ubatch 256 has. the indexer score of qwen4exp is n_kv x n_ubatch x 4 bytes: 4 GiB at 262144 x
rem 4096, 8 GiB at x 8192. an arm "ub ctx" with a smaller ctx makes it smaller. the compute buffer
rem is cut into at most 16 blocks of GGML_VK_SUBALLOCATION_BLOCK_SIZE (1 GiB), the last one takes
rem the rest. see the compute buffer section of README.md and 16-ctx.bat.
rem the server reserves logits for a few tokens only (n_outputs_max in the log), llama-bench and
rem llama-perplexity for every token of a ubatch: 7.6 GiB at ub 8192. the server also runs at the
rem full context and the tools at a short one, so the memory of one does not predict the other.
rem -b has to be at least -ub: the context takes min(n_batch, n_ubatch), and the server default
rem -b is 2048. the n_ubatch line of each arm shows what the context took.
rem the arms inherit every GGML_VK_* and LLAMA_* knob of the environment, LLAMA_MOE_CACHE_STATS
rem and GGML_SCHED_PROF are cleared. the exit code is 1 when no arm survived.
rem note: the arms are torn down with taskkill on llama-server.exe, so the script refuses to start
rem while another llama-server runs.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not defined UBMEMSTATIC  set "UBMEMSTATIC=2048"
if not defined UBMEMARMS    set "UBMEMARMS=8192 16384"
if not defined UBMEMCTX     set "UBMEMCTX=auto"
if not defined UBMEMARGS    set "UBMEMARGS=auto"
if not defined UBMEMFILL    set "UBMEMFILL=3"
if not defined UBMEMNGEN    set "UBMEMNGEN=256"
if not defined UBMEMPORT    set "UBMEMPORT=8098"
if not defined UBMEMWAIT    set "UBMEMWAIT=900"
if not defined UBMEMTIMEOUT set "UBMEMTIMEOUT=1800"
if not defined UBMEMDRAFT   set "UBMEMDRAFT=auto"
if not defined MOEDMONGPU   set "MOEDMONGPU=0"
if not defined SPECNMAX     set "SPECNMAX=6"
if not defined SPECPMIN     set "SPECPMIN=0.6"
if not defined SETTLE       set "SETTLE=30"
if not defined UBMEMPROMPT  set "UBMEMPROMPT=Explain in detail how a modern operating system schedules threads on a multi core processor. Cover run queues, load balancing, priority inheritance and cache affinity."

rem the long prompt is sized by the largest ub of all arms. a list can hold quoted arms, so it is
rem not compared as a whole: none is skipped arm by arm
set "UBMAX=0"
for %%v in (%UBMEMSTATIC% %UBMEMARMS%) do (
    for /f "tokens=1" %%a in ("%%~v") do if /i not "%%a"=="none" if %%a gtr !UBMAX! set "UBMAX=%%a"
)
if "%UBMAX%"=="0" (
    echo no arms: UBMEMSTATIC and UBMEMARMS are none
    exit /b 1
)

rem the context of 14-server
if /i not "%UBMEMCTX%"=="auto" goto :ctx_set
set "UBMEMCTX=%SRVCTX%"
if not defined UBMEMCTX set "UBMEMCTX=262144"
:ctx_set

rem the server setup of models.ini: on cuda all 48 MoE layers on the host and the cache in VRAM
if /i not "%UBMEMARGS%"=="auto" goto :args_set
set "UBMEMARGS=-fit off"
if /i "%BACKEND%"=="cuda" set "UBMEMARGS=-fit off -ncmoe 48 --moe-cache auto"
:args_set

rem the MTP draft of 17, when this machine has one: its context takes memory at the same -ub
if /i not "%UBMEMDRAFT%"=="auto" goto :draft_set
set "UBMEMDRAFT=0"
if defined SPECDRAFT if exist "%SPECDRAFT%" set "UBMEMDRAFT=1"
:draft_set

if not exist "%BIN%\llama-server.exe" (
    echo not built: %BIN%\llama-server.exe
    echo run 00-build.bat first
    exit /b 1
)

if not exist "%MODEL%" (
    echo model not found: %MODEL%
    echo set MODEL in _local.bat
    exit /b 1
)

set "DRAFTARGS="
if not "%UBMEMDRAFT%"=="1" goto :draft_args_set
if not defined SPECDRAFT (
    echo UBMEMDRAFT is 1 but SPECDRAFT is not set
    exit /b 1
)
if not exist "%SPECDRAFT%" (
    echo draft model not found: %SPECDRAFT%
    exit /b 1
)
set "DRAFTARGS=-md "%SPECDRAFT%" --spec-type draft-mtp --spec-draft-n-min 0 --spec-draft-n-max %SPECNMAX% --spec-draft-p-min %SPECPMIN%"
:draft_args_set

where curl.exe >nul 2>&1
if errorlevel 1 (
    echo curl.exe not found in PATH
    exit /b 1
)

tasklist /fi "imagename eq llama-server.exe" 2>nul | find /i "llama-server.exe" >nul
if not errorlevel 1 (
    echo a llama-server.exe is already running
    echo it holds memory, and this script tears its arms down by image name
    exit /b 1
)

set "UBNVSMI=0"
if /i not "%BACKEND%"=="cuda" goto :nvsmi_set
where nvidia-smi >nul 2>&1
if not errorlevel 1 set "UBNVSMI=1"
:nvsmi_set

rem the long prompt is wikitext, without it the short prompt is repeated
if not exist "%PPLFILE%" call "%~dp0get-wikitext.bat"

set "LLAMA_MOE_CACHE_STATS="
set "GGML_SCHED_PROF="

set "WORK=%LOGS%\22-ubmem-%TS%"
if not exist "%WORK%" mkdir "%WORK%"
set "SUM=%LOGS%\22-ubmem-%TS%-summary.txt"
set "TABLE=%WORK%\table.txt"
set "PHASEF=%WORK%\phase.txt"
set "RC=0"
set "FIRST=1"
set "OKLIST="
set "BADLIST="

echo ### 22-ubmem %TS% > "%SUM%"
echo ### MODEL=%MODEL% >> "%SUM%"
echo ### static=%UBMEMSTATIC% phase=%UBMEMARMS% ctx=%UBMEMCTX% fill=%UBMEMFILL% of ub %UBMAX% n_predict=%UBMEMNGEN% >> "%SUM%"
echo ### LOADMODE=%LOADMODE% EXTRA=%EXTRA% UBMEMARGS=%UBMEMARGS% >> "%SUM%"
echo ### draft=%UBMEMDRAFT% %DRAFTARGS% >> "%SUM%"

set "POUT=%WORK%\short.json"
powershell -NoProfile -Command "$j = [ordered]@{ prompt = $env:UBMEMPROMPT; n_predict = [int]$env:UBMEMNGEN; temperature = 0; top_k = 1; ignore_eos = $true; cache_prompt = $false; stream = $false } | ConvertTo-Json -Compress -Depth 5; [System.IO.File]::WriteAllText($env:POUT, $j, (New-Object System.Text.UTF8Encoding($false)))"
if not exist "%POUT%" (
    echo failed to build the request body %POUT%
    exit /b 1
)

for %%v in (%UBMEMSTATIC%) do (
    for /f "tokens=1,2" %%a in ("%%~v") do if /i not "%%a"=="none" call :arm static %%a %%b
)
for %%v in (%UBMEMARMS%) do (
    for /f "tokens=1,2" %%a in ("%%~v") do if /i not "%%a"=="none" call :arm phase %%a %%b
)

echo. >> "%SUM%"
echo ### all arms >> "%SUM%"
if exist "%TABLE%" type "%TABLE%" >> "%SUM%"
if not defined OKLIST echo ### no arm survived >> "%SUM%"
if not defined OKLIST set "RC=1"
if defined OKLIST echo ### survived:%OKLIST% >> "%SUM%"
if defined BADLIST echo ### failed:%BADLIST% >> "%SUM%"

echo.
echo summary in %SUM%
type "%SUM%"
exit /b %RC%

rem ------------------------------------------------------------------
rem %1 = static or phase, %2 = ubatch, %3 = context, empty = UBMEMCTX
:arm
set "ARMMODE=%~1"
set "UB=%~2"
set "CTX=%~3"
if not defined CTX set "CTX=%UBMEMCTX%"
set "PHASEARG="
if "%ARMMODE%"=="phase" set "PHASEARG=--phase-mem"
set "TAG=%ARMMODE%-ub%UB%-c%CTX%"
set "SRVLOG=%LOGS%\22-ubmem-%TS%-%TAG%.log"
set "MLOG=%WORK%\%TAG%-mem.txt"
set "RES=%WORK%\%TAG%-req.txt"
del "%RES%" >nul 2>&1

if "%FIRST%"=="1" (
    set "FIRST=0"
) else (
    echo === waiting %SETTLE%s for the gpu to be released
    powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
)

rem UBMEMFILL full ubatches and a half one of UBMAX at about 4 characters per token, capped at 2
rem characters per token of the context. prompt_n in the summary is the real count
set "POUT=%WORK%\long-%TAG%.json"
powershell -NoProfile -Command "$ub = [int]$env:UBMAX; $n = ([int]$env:UBMEMFILL * $ub + [int][math]::Floor($ub / 2)) * 4; $cx = [int]$env:CTX; if (($cx -gt 0) -and ($n -gt ($cx - 1024) * 2)) { $n = ($cx - 1024) * 2 }; $t = ''; if (Test-Path -LiteralPath $env:PPLFILE) { $t = [IO.File]::ReadAllText($env:PPLFILE) }; if ($t.Length -lt 1000) { $t = $env:UBMEMPROMPT + ' ' }; $sb = New-Object System.Text.StringBuilder; while ($sb.Length -lt $n) { [void]$sb.Append($t) }; $p = $sb.ToString().Substring(0, $n); $j = [ordered]@{ prompt = $p; n_predict = 16; temperature = 0; top_k = 1; ignore_eos = $true; cache_prompt = $false; stream = $false } | ConvertTo-Json -Compress -Depth 5; [System.IO.File]::WriteAllText($env:POUT, $j, (New-Object System.Text.UTF8Encoding($false)))"
if not exist "%POUT%" (
    echo failed to build the request body %POUT%
    set "RC=1"
    goto :eof
)

set "SARGS=-m "%MODEL%" -c %CTX% -np 1 -b %UB% -ub %UB% -fa on %PHASEARG% %LOADMODE% %EXTRA% %UBMEMARGS% %DRAFTARGS% -lv 4 --host 127.0.0.1 --port %UBMEMPORT%"
echo === %ARMMODE% ub %UB% ctx %CTX% -^> %SRVLOG%
> "%PHASEF%" echo load
call :mem_start
set "SCMD=%WORK%\srv-%TAG%.cmd"
> "%SCMD%" echo @echo off
>>"%SCMD%" echo "%BIN%\llama-server.exe" %SARGS% ^> "%SRVLOG%" 2^>^&1
start "22-ubmem-%TAG%" /min cmd /c "%SCMD%"

set "STATE=survived"
call :waitup
if "%UP%"=="1" goto :arm_up
set "STATE=failed at load"
if not "%GONE%"=="1" set "STATE=no health after %WAITED%s"
echo llama-server did not come up, last lines of %SRVLOG%:
powershell -NoProfile -Command "Get-Content -Tail 20 -LiteralPath $env:SRVLOG"
goto :arm_down
:arm_up
for %%r in (short1 long short2) do if "!STATE!"=="survived" call :req %%r
if not "%STATE%"=="survived" goto :arm_down
set "CODE="
curl.exe -s -o nul -w "%%{http_code}" "http://127.0.0.1:%UBMEMPORT%/health" > "%WORK%\health.txt" 2>nul
if exist "%WORK%\health.txt" set /p CODE=<"%WORK%\health.txt"
if not "%CODE%"=="200" set "STATE=no health after short2"
:arm_down
> "%PHASEF%" echo end
rem the slot prints its timings as it is released, give it a moment before the process goes away
powershell -NoProfile -Command "Start-Sleep -Seconds 2"
call :mem_stop
call :teardown

set "HASCACHE=0"
findstr /c:"slots per layer" "%SRVLOG%" >nul 2>&1
if not errorlevel 1 set "HASCACHE=1"

echo. >> "%SUM%"
echo ### %ARMMODE% ub %UB% ctx %CTX%: %STATE% >> "%SUM%"
if exist "%RES%" type "%RES%" >> "%SUM%"
call :log_sum
call :mem_sum
call :arm_row

if "%STATE%"=="survived" (
    set "OKLIST=%OKLIST% %ARMMODE%-%UB%/%CTX%"
) else (
    set "BADLIST=%BADLIST% %ARMMODE%-%UB%/%CTX%=%STATE%;"
)
echo === %ARMMODE% ub %UB% ctx %CTX%: %STATE%
goto :eof

rem ------------------------------------------------------------------
rem %1 = phase: short1, long or short2. no 200 with timings, or no server after it, ends the arm
:req
set "PHASE=%~1"
> "%PHASEF%" echo %PHASE%
set "BODY=%WORK%\short.json"
if "%PHASE%"=="long" set "BODY=%WORK%\long-%TAG%.json"
set "RFILE=%WORK%\%TAG%-%PHASE%-resp.json"
set "HFILE=%WORK%\%TAG%-%PHASE%-http.txt"
set "CFILE=%WORK%\%TAG%-%PHASE%-curl.txt"
del "%RFILE%" >nul 2>&1
echo === %ARMMODE% ub %UB%: %PHASE%
curl.exe -s -S --max-time %UBMEMTIMEOUT% -H "Content-Type: application/json" --data-binary @"%BODY%" -o "%RFILE%" -w "%%{http_code}" "http://127.0.0.1:%UBMEMPORT%/completion" > "%HFILE%" 2> "%CFILE%"
set "CURLRC=%ERRORLEVEL%"
powershell -NoProfile -Command "$c = [Globalization.CultureInfo]::InvariantCulture; $code = ''; if (Test-Path -LiteralPath $env:HFILE) { $code = ([string](Get-Content -Raw -LiteralPath $env:HFILE)).Trim() }; $j = $null; try { $j = Get-Content -Raw -LiteralPath $env:RFILE -ErrorAction Stop | ConvertFrom-Json } catch { }; if (($code -eq '200') -and ($null -ne $j) -and ($null -ne $j.timings)) { $t = $j.timings; Add-Content -LiteralPath $env:RES -Value ('{0} ok http={1} prompt_n={2} pp={3} t/s predicted_n={4} tg={5} t/s' -f $env:PHASE, $code, $t.prompt_n, ([math]::Round([double]$t.prompt_per_second, 2)).ToString($c), $t.predicted_n, ([math]::Round([double]$t.predicted_per_second, 2)).ToString($c)); exit 0 }; $m = ''; if (($null -ne $j) -and ($null -ne $j.error)) { $m = [string]$j.error.message }; Add-Content -LiteralPath $env:RES -Value ('{0} fail http={1} curl={2} {3}' -f $env:PHASE, $code, $env:CURLRC, $m); exit 1"
set "REQRC=%ERRORLEVEL%"
rem a crashed server can take a moment to leave the process list
if not "%REQRC%"=="0" powershell -NoProfile -Command "Start-Sleep -Seconds 2"
tasklist /fi "imagename eq llama-server.exe" 2>nul | find /i "llama-server.exe" >nul
if errorlevel 1 (
    set "STATE=died in %PHASE%"
    goto :eof
)
if not "%REQRC%"=="0" set "STATE=failed in %PHASE%"
goto :eof

rem ------------------------------------------------------------------
rem one line per sample: time of day in ms, phase, process dedicated and shared, adapter
rem dedicated and shared, nvidia-smi used (-1 = none), process private bytes, all in MiB. the
rem adapter is the one the process uses, before the process has one the busiest adapter
:mem_start
set "MRUN=%MLOG%.run"
set "MDONE=%MLOG%.done"
del "%MDONE%" >nul 2>&1
del "%MLOG%" >nul 2>&1
echo.> "%MRUN%"
start "" /b powershell -NoProfile -Command "$stop = (Get-Date).AddHours(6); $ph = 'start'; $lu = $null; $gp = 'Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory'; $ga = 'Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory'; while ((Test-Path -LiteralPath $env:MRUN) -and ((Get-Date) -lt $stop)) { $s = Get-Date; try { $x = Get-Content -LiteralPath $env:PHASEF -TotalCount 1 -ErrorAction Stop; if ($x) { $ph = ([string]$x).Trim() } } catch { }; $pd = 0; $psh = 0; $ad = 0; $ash = 0; $pm = 0; $nu = -1; $p = Get-Process -Name llama-server -ErrorAction SilentlyContinue | Select-Object -First 1; if ($p) { $pm = [math]::Round($p.PrivateMemorySize64 / 1MB); $g = @(Get-CimInstance -ClassName $gp -ErrorAction SilentlyContinue | Where-Object { $_.Name -like ('pid_' + $p.Id + '_*') } | Sort-Object { [double]$_.DedicatedUsage } -Descending); if ($g.Count -gt 0) { $pd = [math]::Round([double]$g[0].DedicatedUsage / 1MB); $psh = [math]::Round([double]$g[0].SharedUsage / 1MB); $lu = $g[0].Name.Substring($g[0].Name.IndexOf('luid_')) } }; $a = @(Get-CimInstance -ClassName $ga -ErrorAction SilentlyContinue); $sel = $null; if ($lu) { $sel = $a | Where-Object { $_.Name -eq $lu } | Select-Object -First 1 }; if (-not $sel) { $sel = $a | Sort-Object { [double]$_.DedicatedUsage } -Descending | Select-Object -First 1 }; if ($sel) { $ad = [math]::Round([double]$sel.DedicatedUsage / 1MB); $ash = [math]::Round([double]$sel.SharedUsage / 1MB) }; if ($env:UBNVSMI -eq '1') { $o = nvidia-smi '--query-gpu=memory.used' '--format=csv,noheader,nounits' -i $env:MOEDMONGPU 2>$null; $v = 0; if ([int]::TryParse(([string]$o).Trim(), [ref]$v)) { $nu = $v } }; $t = [int][math]::Floor((Get-Date).TimeOfDay.TotalMilliseconds); Add-Content -LiteralPath $env:MLOG -Value ('{0} {1} {2} {3} {4} {5} {6} {7}' -f $t, $ph, $pd, $psh, $ad, $ash, $nu, $pm); $w = 1000 - ((Get-Date) - $s).TotalMilliseconds; if ($w -gt 50) { Start-Sleep -Milliseconds ([int]$w) } }; Set-Content -LiteralPath $env:MDONE -Value done"
goto :eof

:mem_stop
del "%MRUN%" >nul 2>&1
powershell -NoProfile -Command "$n = 0; while ((-not (Test-Path -LiteralPath $env:MDONE)) -and ($n -lt 30)) { Start-Sleep -Milliseconds 500; $n++ }"
del "%MDONE%" >nul 2>&1
goto :eof

rem ------------------------------------------------------------------
rem the buffers of the load, up to the listening line, then the phase switches, the cache
rem allocations and the errors. a --phase-mem arm reserves again at every switch, those buffer
rem lines stay out, the switch lines carry their sizes
:log_sum
powershell -NoProfile -Command "if (-not (Test-Path -LiteralPath $env:SRVLOG)) { exit 0 }; $pre = $true; foreach ($l in (Get-Content -LiteralPath $env:SRVLOG)) { if ($pre -and ($l -match 'listening on ')) { $pre = $false }; if ($pre -and ($l -match 'n_ubatch|n_outputs_max |phase_mem  |model buffer size|KV buffer size|RS buffer size|output buffer size|compute buffer size|splits = ')) { $l } elseif ($l -match 'moe_cache: |phase_mem: prompt|phase_mem: generation|ubatch does not fit|trying a smaller one|phase_mem needs|Device memory allocation of size|exceeds device buffer size|No suitable memory type found|not enough space in the buffer|failed to allocate|CUDA error|out of memory|GGML_ASSERT') { $l } }" >> "%SUM%"
goto :eof

rem ------------------------------------------------------------------
rem the maximum of every column per phase, the growth from short1 to the phases after it, and
rem the decode speed of short2 against short1. the device column is nvidia-smi when there is one
:mem_sum
powershell -NoProfile -Command "$ord = New-Object System.Collections.ArrayList; $mx = @{}; $cnt = @{}; if (Test-Path -LiteralPath $env:MLOG) { foreach ($l in (Get-Content -LiteralPath $env:MLOG)) { $f = $l.Trim() -split '\s+'; if ($f.Count -lt 8) { continue }; $ph = $f[1]; if (-not $mx.ContainsKey($ph)) { [void]$ord.Add($ph); $mx[$ph] = @(0.0, 0.0, 0.0, 0.0, -1.0, 0.0); $cnt[$ph] = 0 }; $cnt[$ph] = $cnt[$ph] + 1; for ($i = 0; $i -lt 6; $i++) { $x = [double]$f[$i + 2]; if ($x -gt $mx[$ph][$i]) { $mx[$ph][$i] = $x } } } }; if ($ord.Count -eq 0) { 'memory: no samples in ' + $env:MLOG; exit 0 }; 'memory, MiB, max per phase: samples | process dedicated, shared | adapter dedicated, shared | nvidia-smi | process private'; foreach ($ph in $ord) { $m = $mx[$ph]; '  {0,-7} {1,5} | {2,7} {3,7} | {4,7} {5,7} | {6,7} | {7,7}' -f $ph, $cnt[$ph], $m[0], $m[1], $m[2], $m[3], $m[4], $m[5] }; if (-not $mx.ContainsKey('short1')) { exit 0 }; $k = 0; $src = 'process dedicated'; if ($mx['short1'][4] -ge 0) { $k = 4; $src = 'nvidia-smi' }; $da = -1.0; $sa = -1.0; foreach ($ph in @('long', 'short2')) { if ($mx.ContainsKey($ph)) { if ($mx[$ph][$k] -gt $da) { $da = $mx[$ph][$k] }; if ($mx[$ph][1] -gt $sa) { $sa = $mx[$ph][1] } } }; if ($da -lt 0) { exit 0 }; $g = $da - $mx['short1'][$k]; $gs = $sa - $mx['short1'][1]; 'growth after short1: ' + $src + ' ' + $g + ' MiB, process shared ' + $gs + ' MiB'; if (($env:HASCACHE -eq '1') -and ($g -ge 900)) { 'FLAG: after the cache took its slots the device use grew by ' + $g + ' MiB of the 1024 MiB it leaves free' }; if ($gs -gt 512) { 'FLAG: shared memory grew by ' + $gs + ' MiB after short1: an allocation went to host memory instead of failing' }; $inv = [Globalization.CultureInfo]::InvariantCulture; $r = @{}; if (Test-Path -LiteralPath $env:RES) { foreach ($l in (Get-Content -LiteralPath $env:RES)) { if ($l -match '^(\S+) ok .* tg=([0-9.]+) t/s') { $r[$matches[1]] = [double]::Parse($matches[2], $inv) } } }; if ($r.ContainsKey('short1') -and $r.ContainsKey('short2') -and ($r['short1'] -gt 0)) { $q = $r['short2'] / $r['short1']; 'decode short2 / short1: ' + ([math]::Round($q, 3)).ToString($inv); if ($q -lt 0.8) { 'FLAG: the decode after the long prefill is more than 20 percent slower than before it' } }" >> "%SUM%"
goto :eof

rem ------------------------------------------------------------------
rem one row of the table at the end. splits are the first ones of the log, the reserve of the
rem target model at the load; the draft context prints its own after them. a compute buffer is
rem "first..max" when a later reserve made it larger
:arm_row
powershell -NoProfile -Command "$c = [Globalization.CultureInfo]::InvariantCulture; $r = @{}; if (Test-Path -LiteralPath $env:RES) { foreach ($l in (Get-Content -LiteralPath $env:RES)) { $f = $l -split ' '; if ($f.Count -gt 1) { $r[$f[0]] = $l } } }; $v = { param($k, $n) if ($r.ContainsKey($k) -and ($r[$k] -match (' ' + $n + '=([0-9.]+)'))) { $matches[1] } else { '-' } }; $sp = '-'; $sl = New-Object System.Collections.ArrayList; $pu = New-Object System.Collections.ArrayList; $swn = 0; $sws = 0.0; $cb = New-Object System.Collections.ArrayList; $c1 = @{}; $cm = @{}; if (Test-Path -LiteralPath $env:SRVLOG) { foreach ($l in (Get-Content -LiteralPath $env:SRVLOG)) { if (($sp -eq '-') -and ($l -match 'splits = (\d+)')) { $sp = $matches[1] }; if ($l -match '(\d+) slots per layer') { [void]$sl.Add([int]$matches[1]) }; if ($l -match 'moe_cache: .* in ([0-9.]+) ms$') { $sws += [double]::Parse($matches[1], $c) / 1000 }; if ($l -match 'phase_mem: (prompt|generation), .* ([0-9.]+) s$') { $swn++; $sws += [double]::Parse($matches[2], $c) }; if ($l -match 'phase_mem: prompt, .* ubatch (\d+) of ') { [void]$pu.Add([int]$matches[1]) }; if ($l -match '(\S+) compute buffer size = +([0-9.]+) MiB') { $b = $matches[1]; $x = [math]::Round([double]::Parse($matches[2], $c)); if (-not $c1.ContainsKey($b)) { [void]$cb.Add($b); $c1[$b] = $x; $cm[$b] = $x } elseif ($x -gt $cm[$b]) { $cm[$b] = $x } } } }; $sls = '-'; if ($sl.Count -gt 0) { $sls = [string]$sl[0]; if ($sl[$sl.Count - 1] -ne $sl[0]) { $sls = $sls + '/' + $sl[$sl.Count - 1] } }; $pus = $env:UB; if ($env:ARMMODE -eq 'phase') { $pus = '-'; if ($pu.Count -gt 0) { $mm = $pu | Measure-Object -Minimum -Maximum; $pus = [string]$mm.Minimum; if ($mm.Maximum -gt $mm.Minimum) { $pus = $pus + '..' + $mm.Maximum } } }; $cbs = ($cb | ForEach-Object { if ($cm[$_] -gt $c1[$_]) { $_ + ' ' + $c1[$_] + '..' + $cm[$_] } else { $_ + ' ' + $c1[$_] } }) -join ', '; $dm = 0; $sm = 0; if (Test-Path -LiteralPath $env:MLOG) { foreach ($l in (Get-Content -LiteralPath $env:MLOG)) { $f = $l.Trim() -split '\s+'; if ($f.Count -lt 8) { continue }; $d = [double]$f[2]; if ([double]$f[6] -ge 0) { $d = [double]$f[6] }; if ($d -gt $dm) { $dm = $d }; if ([double]$f[3] -gt $sm) { $sm = [double]$f[3] } } }; $fl = ''; $f0 = Join-Path $env:WORK 'splits0.txt'; if (($env:ARMMODE -eq 'static') -and ($sp -ne '-')) { if (Test-Path -LiteralPath $f0) { $s0 = [int]([string](Get-Content -LiteralPath $f0 -TotalCount 1)).Trim(); if ([int]$sp -gt $s0) { $fl = $fl + '  FLAG: ' + $sp + ' splits against ' + $s0 + ' in the first static arm, an op went to another backend' } } else { Set-Content -LiteralPath $f0 -Value $sp } }; if (($sl.Count -gt 1) -and ($sl[$sl.Count - 1] -lt $sl[0])) { $fl = $fl + '  FLAG: the cache had ' + $sl[0] + ' slots at first and ' + $sl[$sl.Count - 1] + ' at the end, less memory was free: a pool grew' }; $fmt = '{0,-6} {1,6} {2,7} | {3,-24} | {4,7} {5,8} | {6,7} {7,7} | {8,6} | {9,7} | {10,11} | {11,4} {12,6} | {13,7} {14,7} | {15}'; if (-not (Test-Path -LiteralPath $env:TABLE)) { Add-Content -LiteralPath $env:TABLE -Value ($fmt -f 'mode', 'ub', 'ctx', 'state', 'long_n', 'long_pp', 'tg_s1', 'tg_s2', 'splits', 'slots', 'pp_ub', 'sw_n', 'sw_s', 'dev_max', 'shr_max', 'compute buffers, MiB') }; $row = ($fmt -f $env:ARMMODE, $env:UB, $env:CTX, $env:STATE, (& $v 'long' 'prompt_n'), (& $v 'long' 'pp'), (& $v 'short1' 'tg'), (& $v 'short2' 'tg'), $sp, $sls, $pus, $swn, ([math]::Round($sws, 2)).ToString($c), $dm, $sm, $cbs) + $fl; Add-Content -LiteralPath $env:TABLE -Value $row; $row" >> "%SUM%"
goto :eof

rem ------------------------------------------------------------------
rem poll /health until it answers 200 or UBMEMWAIT runs out. the sleep comes first: the process
rem needs a moment to appear in tasklist, and checking before that would call the arm dead
:waitup
set "UP=0"
set "GONE=0"
set "WAITED=0"
:waitup_loop
powershell -NoProfile -Command "Start-Sleep -Seconds 3"
set /a "WAITED+=3"
set "CODE="
curl.exe -s -o nul -w "%%{http_code}" "http://127.0.0.1:%UBMEMPORT%/health" > "%WORK%\health.txt" 2>nul
if exist "%WORK%\health.txt" set /p CODE=<"%WORK%\health.txt"
if "%CODE%"=="200" (
    set "UP=1"
    goto :eof
)
tasklist /fi "imagename eq llama-server.exe" 2>nul | find /i "llama-server.exe" >nul
if errorlevel 1 (
    set "GONE=1"
    goto :eof
)
if %WAITED% geq %UBMEMWAIT% goto :eof
goto :waitup_loop

rem ------------------------------------------------------------------
:teardown
taskkill /f /im llama-server.exe >nul 2>&1
goto :eof
