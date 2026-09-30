@echo off
rem speculative decoding off, ngram-mod, draft-mtp and both, with fixed draft lengths and with
rem --spec-auto, one llama-server start per arm, -np 1. an arm is "name types [args]":
rem   types none   no speculation and no draft model, the baseline
rem   types infer  the draft model without --spec-type: the type comes from the gguf, and with
rem                --spec-auto ngram-mod joins it. this is the fully automatic setup
rem   other types  go to --spec-type as given; the draft model is loaded when they name a draft-* type
rem --spec-auto picks the draft length on every cycle: a draft position stays while the chance that
rem it and all positions before it are accepted is above R * c, where R is the measured token rate
rem and c the measured cost of one more verified token (plus one draft step for draft-mtp). the
rem chance comes from the draft head probability, calibrated online against the acceptance, and for
rem ngram-mod from its acceptance per position. --spec-draft-n-max stays as the cap, 6 when it is not
rem given.
rem every arm runs the same requests with sampling on, the setup of models.ini (temperature 1.0,
rem top-k 20, top-p 0.95): SPECAUTOPROMPTS x SPECAUTOSEEDS chat requests of up to SPECAUTONGEN
rem tokens, after one short warmup request that is not counted. with sampling the text differs
rem between the arms, so the arm total is sum(predicted_n) / sum(predicted_ms) over all requests,
rem and the per-cycle trace is kept for the offline comparison:
rem LLAMA_SPEC_TRACE=<file> writes one line per draft cycle - draft length, accepted count, cycle
rem and draft time, head probability per position, target probability of the drafted token, the
rem top target probability, and the acceptance exact speculative sampling would give.
rem the prompts:
rem   os      explain thread scheduling, plain prose
rem   db      explain join execution, plain prose
rem   code    write a C++ class, code
rem   repeat  write back a generated block verbatim, the case ngram-mod is built for
rem note: the arms are torn down with taskkill on llama-server.exe, so this stops any other
rem llama-server running on the machine. the script refuses to start if one is already up.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not defined SPECAUTOARMS    set "SPECAUTOARMS="none none" "n3 draft-mtp --spec-draft-n-max 3" "auto infer --spec-auto""
if not defined SPECAUTOPROMPTS set "SPECAUTOPROMPTS=os db code repeat"
if not defined SPECAUTOSEEDS   set "SPECAUTOSEEDS=1 2"
if not defined SPECAUTONGEN    set "SPECAUTONGEN=1024"
if not defined SPECAUTOCTX     set "SPECAUTOCTX=auto"
if not defined SPECAUTOARGS    set "SPECAUTOARGS=auto"
if not defined SPECAUTOSAMP    set "SPECAUTOSAMP=1.0 20 0.95"
if not defined SPECAUTOPORT    set "SPECAUTOPORT=8097"
if not defined SPECAUTOWAIT    set "SPECAUTOWAIT=900"
if not defined SPECAUTOTIMEOUT set "SPECAUTOTIMEOUT=1800"
if not defined SPECREPLINE     set "SPECREPLINE=80"
if not defined SETTLE          set "SETTLE=30"

if /i not "%SPECAUTOCTX%"=="auto" goto :ctx_set
set "SPECAUTOCTX=%SRVCTX%"
if not defined SPECAUTOCTX set "SPECAUTOCTX=262144"
:ctx_set

rem auto is the setup of models.ini
if /i not "%SPECAUTOARGS%"=="auto" goto :args_set
set "SPECAUTOARGS=-fit off"
if /i "%BACKEND%"=="cuda" set "SPECAUTOARGS=-fit off -ncmoe 48 --moe-cache auto"
:args_set

for /f "tokens=1,2,3" %%a in ("%SPECAUTOSAMP%") do (
    set "STEMP=%%a"
    set "STOPK=%%b"
    set "STOPP=%%c"
)

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

if not defined SPECDRAFT (
    echo SPECDRAFT is not set
    echo put the path of the MTP draft gguf in _local.bat
    exit /b 1
)

if not exist "%SPECDRAFT%" (
    echo draft model not found: %SPECDRAFT%
    exit /b 1
)

where curl.exe >nul 2>&1
if errorlevel 1 (
    echo curl.exe not found in PATH
    exit /b 1
)

tasklist /fi "imagename eq llama-server.exe" 2>nul | find /i "llama-server.exe" >nul
if not errorlevel 1 (
    echo a llama-server.exe is already running
    echo this script tears its arms down by image name and would stop that one too
    exit /b 1
)

set "WORK=%LOGS%\23-spec-auto-%TS%"
if not exist "%WORK%" mkdir "%WORK%"
set "SUM=%LOGS%\23-spec-auto-%TS%-summary.txt"
set "RC=0"

echo ### 23-spec-auto %TS% > "%SUM%"
echo ### MODEL=%MODEL% >> "%SUM%"
echo ### SPECDRAFT=%SPECDRAFT% >> "%SUM%"
echo ### ctx=%SPECAUTOCTX% n_predict=%SPECAUTONGEN% prompts=%SPECAUTOPROMPTS% seeds=%SPECAUTOSEEDS% temp=%STEMP% top_k=%STOPK% top_p=%STOPP% >> "%SUM%"
echo ### LOADMODE=%LOADMODE% EXTRA=%EXTRA% SPECAUTOARGS=%SPECAUTOARGS% >> "%SUM%"
echo. >> "%SUM%"

call :mkreq warmup
for %%p in (%SPECAUTOPROMPTS%) do call :mkreq %%p
if not "%RC%"=="0" exit /b 1

set "ABORT=0"
for %%a in (%SPECAUTOARMS%) do call :arm "%%~a"

set "LLAMA_SPEC_TRACE="

echo.
echo summary in %SUM%
type "%SUM%"
exit /b %RC%

rem ------------------------------------------------------------------
rem %1 = prompt name. powershell writes the chat body, so the prompt does not have to survive batch
rem quoting, and WriteAllText keeps the utf8 BOM out of it. the seed goes in per request, see :req
:mkreq
set "PNAME=%~1"
set "PLINES=%SPECREPLINE%"
set "POUT=%WORK%\prompt-%PNAME%.txt"
powershell -NoProfile -Command "$n = $env:PNAME; if ($n -eq 'warmup') { $p = 'Describe your favourite season in a short paragraph.' } elseif ($n -eq 'os') { $p = 'Explain in detail how a modern operating system schedules threads on a multi core processor. Cover run queues, load balancing, priority inheritance and cache affinity.' } elseif ($n -eq 'db') { $p = 'Explain in detail how a relational database executes a join between two large tables. Cover hash join, merge join, nested loops, spilling to disk and cardinality estimation.' } elseif ($n -eq 'code') { $p = 'Write a C++17 header-only thread-safe LRU cache class template with get, put and erase, then a short usage example and unit tests with assert.' } elseif ($n -eq 'repeat') { $body = [string]::Join([string][char]10, (1..[int]$env:PLINES | ForEach-Object { '    const int row_{0:d3} = accumulate(buffer, stride * {0}, offset + {0}, mask_{0:d3});' -f $_ })); $p = 'Write the following block of text back to me, exactly as it is. Output only the block, no commentary and no code fences.' + [string][char]10 + [string][char]10 + $body } else { exit 1 }; [System.IO.File]::WriteAllText($env:POUT, $p, (New-Object System.Text.UTF8Encoding($false)))"
if not exist "%POUT%" (
    echo unknown prompt %PNAME%, known: os db code repeat
    set "RC=1"
)
goto :eof

rem ------------------------------------------------------------------
rem %1 = "name types [args]"
:arm
if "%ABORT%"=="1" goto :eof
set "ARM="
set "STYPE="
set "AARGS="
for /f "tokens=1,2,*" %%a in ("%~1") do (
    set "ARM=%%a"
    set "STYPE=%%b"
    set "AARGS=%%c"
)
if not defined STYPE (
    echo bad arm "%~1", expected "name types [args]"
    set "RC=1"
    goto :eof
)

set "SPECT=--spec-type %STYPE%"
set "SPECMD="
if /i "%STYPE%"=="none" set "SPECT="
if /i "%STYPE%"=="infer" set "SPECT="
if /i "%STYPE%"=="infer" set "SPECMD=-md "%SPECDRAFT%""
if not "!STYPE:draft-=!"=="!STYPE!" set "SPECMD=-md "%SPECDRAFT%""

set "SRVLOG=%LOGS%\23-spec-auto-%TS%-%ARM%.log"
set "LLAMA_SPEC_TRACE=%WORK%\trace-%ARM%.csv"
set "SARGS=-m "%MODEL%" %SPECMD% %SPECT% %AARGS% -c %SPECAUTOCTX% -np 1 -fa on --repeat-penalty 1.0 %LOADMODE% %EXTRA% %SPECAUTOARGS% --host 127.0.0.1 --port %SPECAUTOPORT% -lv 4"

echo === %ARM%: types %STYPE% %AARGS% -^> %SRVLOG%
set "SCMD=%WORK%\srv-%ARM%.cmd"
> "%SCMD%" echo @echo off
>>"%SCMD%" echo "%BIN%\llama-server.exe" %SARGS% ^> "%SRVLOG%" 2^>^&1
start "23-spec-auto-%ARM%" /min cmd /c "%SCMD%"

call :waitup
if not "%UP%"=="1" (
    if "%GONE%"=="1" (
        set "FMSG=llama-server.exe exited after %WAITED%s"
    ) else (
        set "FMSG=no 200 from /health after %WAITED%s"
    )
    echo !FMSG!, last lines of %SRVLOG%:
    powershell -NoProfile -Command "Get-Content -Tail 20 -LiteralPath $env:SRVLOG"
    echo. >> "%SUM%"
    echo ### %ARM%: !FMSG!, the next arms are skipped, see %SRVLOG% >> "%SUM%"
    set "RC=1"
    set "ABORT=1"
    call :teardown
    goto :eof
)

rem the warmup also covers the first 64 cycles of --spec-auto, which probe draft lengths. a request
rem error here is the same for every arm, so the run stops
call :req warmup 0 256
if "%RERR%"=="1" (
    echo the warmup request failed, the next arms are skipped
    echo. >> "%SUM%"
    echo ### %ARM%: the warmup request failed, the next arms are skipped, see %RRESP% >> "%SUM%"
    set "ABORT=1"
    call :teardown
    goto :eof
)
for %%p in (%SPECAUTOPROMPTS%) do (
    for %%s in (%SPECAUTOSEEDS%) do call :req %%p %%s %SPECAUTONGEN%
)

rem the slot prints its timings as it is released, give it a moment before the process goes away
powershell -NoProfile -Command "Start-Sleep -Seconds 2"
call :teardown

rem the arm total over every counted request: sum of tokens over sum of eval time. InvariantCulture
rem keeps a russian locale from printing 16,842
set "AWORK=%WORK%"
set "AARM=%ARM%"
echo. >> "%SUM%"
echo ### %ARM%: types %STYPE% %AARGS% >> "%SUM%"
powershell -NoProfile -Command "$c = [System.Globalization.CultureInfo]::InvariantCulture; $n = 0; $ms = 0; $dn = 0; $da = 0; $rows = @(); Get-ChildItem -LiteralPath $env:AWORK -Filter ($env:AARM + '-*-resp.json') | Where-Object { $_.Name -notlike '*-warmup-*' } | Sort-Object Name | ForEach-Object { $f = $_.Name; try { $r = Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json; $t = $r.timings; if ($null -eq $t) { throw [string]$r.error.message }; $n += $t.predicted_n; $ms += $t.predicted_ms; $dn += $t.draft_n; $da += $t.draft_n_accepted; $rows += ('    {0}: {1} tokens, {2} t/s, draft {3}/{4}' -f $_.Name, $t.predicted_n, [math]::Round($t.predicted_per_second, 2).ToString($c), $t.draft_n_accepted, $t.draft_n) } catch { $rows += ('    {0}: no timings: {1}' -f $f, $_.Exception.Message) } }; $rows | ForEach-Object { $_ }; $tps = if ($ms -gt 0) { 1000.0 * $n / $ms } else { 0 }; $acc = if ($dn -gt 0) { $da / $dn } else { 0 }; '### total {0} tokens in {1} ms = {2} t/s, draft acceptance {3} ({4} / {5})' -f $n, [math]::Round($ms, 1).ToString($c), [math]::Round($tps, 2).ToString($c), [math]::Round($acc, 4).ToString($c), $da, $dn" >> "%SUM%"
findstr /c:"auto:" /c:"statistics " "%SRVLOG%" >> "%SUM%"

echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
goto :eof

rem ------------------------------------------------------------------
rem %1 = prompt name, %2 = seed, %3 = max tokens
:req
set "RNAME=%~1"
set "RSEED=%~2"
set "RN=%~3"
set "RERR=0"
set "RPROMPT=%WORK%\prompt-%RNAME%.txt"
set "RBODY=%WORK%\%ARM%-%RNAME%-s%RSEED%-req.json"
set "RRESP=%WORK%\%ARM%-%RNAME%-s%RSEED%-resp.json"
set "RTEMP=%STEMP%"
set "RTOPK=%STOPK%"
set "RTOPP=%STOPP%"
powershell -NoProfile -Command "$c = [System.Globalization.CultureInfo]::InvariantCulture; $p = [IO.File]::ReadAllText($env:RPROMPT); $j = [ordered]@{ messages = @(@{ role = 'user'; content = $p }); max_tokens = [int]$env:RN; temperature = [double]::Parse($env:RTEMP, $c); top_k = [int]$env:RTOPK; top_p = [double]::Parse($env:RTOPP, $c); seed = [int]$env:RSEED; cache_prompt = $false; stream = $false } | ConvertTo-Json -Compress -Depth 5; [System.IO.File]::WriteAllText($env:RBODY, $j, (New-Object System.Text.UTF8Encoding($false)))"
echo     %RNAME% seed %RSEED%
curl.exe -s -S --max-time %SPECAUTOTIMEOUT% -H "Content-Type: application/json" --data-binary @"%RBODY%" -o "%RRESP%" "http://127.0.0.1:%SPECAUTOPORT%/v1/chat/completions" 2> "%WORK%\%ARM%-%RNAME%-s%RSEED%-curl.txt"
if errorlevel 1 (
    echo curl failed on %ARM% %RNAME% seed %RSEED%, see %WORK%\%ARM%-%RNAME%-s%RSEED%-curl.txt
    set "RC=1"
    set "RERR=1"
    goto :eof
)
findstr /c:"predicted_per_second" "%RRESP%" >nul 2>&1
if errorlevel 1 (
    echo no timings in the answer to %ARM% %RNAME% seed %RSEED%:
    type "%RRESP%"
    echo.
    set "RC=1"
    set "RERR=1"
)
goto :eof

rem ------------------------------------------------------------------
rem poll /health until it answers 200 or SPECAUTOWAIT runs out. the sleep comes first: the process
rem needs a moment to appear in tasklist, and checking before that would call the arm dead
:waitup
set "UP=0"
set "GONE=0"
set "WAITED=0"
:waitup_loop
powershell -NoProfile -Command "Start-Sleep -Seconds 3"
set /a "WAITED+=3"
set "CODE="
curl.exe -s -o nul -w "%%{http_code}" "http://127.0.0.1:%SPECAUTOPORT%/health" > "%WORK%\health.txt" 2>nul
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
if %WAITED% geq %SPECAUTOWAIT% goto :eof
goto :waitup_loop

rem ------------------------------------------------------------------
:teardown
taskkill /f /im llama-server.exe >nul 2>&1
goto :eof
