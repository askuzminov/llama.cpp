@echo off
rem does -np help speculative decoding, one llama-server start per arm.
rem the draft side already batches every drafting sequence into one decode, but the target side
rem goes through split_equal with n_keep_tail = 1 + n_rs_seq, and that keeps several slots in one
rem ubatch only when they all carry the same number of tokens. one token of difference in the
rem draft length sends each slot into its own ubatch, so the fixed cost of a target pass is paid
rem twice. the arms separate the two cases:
rem   np1          one slot, one prompt - the baseline
rem   np2-one      two slots, one prompt - what -np costs when the second slot is idle
rem   np2-lock     two slots fed the same prompt twice - both write the same text at temperature 0,
rem                so the draft lengths always match and the ubatch merges. the upper bound
rem   np2-free     two slots, two different prompts - the draft lengths diverge and the ubatch
rem                splits. the realistic case, and np2-lock minus np2-free is the cost of the split
rem   np1-noreuse  one slot with LLAMA_GRAPH_REUSE_DISABLE=1 - the target rebuilds its graph on
rem                every pass, which prices the rebuild that a varying draft length already forces
rem the last five arms answer a different question: does ngram-mod pay off next to draft-mtp. the
rem priority list in common/speculative.cpp puts every ngram speculator above every model one, so
rem ngram-mod drafts first and the mtp head only runs where ngram found nothing. all five run on
rem one slot:
rem   ngram-off    draft-mtp alone on a repeat-verbatim prompt - the baseline for ngram-on
rem   ngram-on     ngram-mod,draft-mtp on the same prompt - what a long verbatim repeat is worth
rem   ngram-crlf   the same block with CRLF line ends - the model writes LF, so the tokens differ
rem                and the pool never matches
rem   ngram-crlf-off  the CRLF block with draft-mtp alone - the baseline ngram-crlf needs, since
rem                the CRLF block also changes what the model writes
rem   ngram-miss   ngram-mod,draft-mtp on the plain prompt, which has nothing to repeat - what the
rem                speculator costs when it never hits
rem the two-slot arms send both prompts in one request, as a json array. the server turns an array
rem into one task per element and posts them together, so both slots start in the same batch. in
rem np2-lock the two prompts are the same string, so they also hold the same token count and
rem split_equal keeps them in one prefill ubatch - both slots leave the prompt in the same decode
rem and stay aligned. two separate curl calls do not give that.
rem note: `n_cmpl` (OpenAI `n`) would be the shorter way to ask for two completions, but its child
rem slot copies only the target memory; draft-mtp has no get_state/set_state, so the child drafts
rem from an empty draft context and acceptance collapses. that is why the prompt is repeated.
rem every arm runs at temperature 0 with ignore_eos and a fixed n_predict, so all of them generate
rem the same number of tokens and the text is deterministic. without that the arms are not
rem comparable: sampling makes each run write different text and the spread reaches 7 percent.
rem note: the arms are torn down with taskkill on llama-server.exe, so this stops any other
rem llama-server running on the machine. the script refuses to start if one is already up.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not defined SPECCTX     set "SPECCTX=32768"
if not defined SPECNMAX    set "SPECNMAX=6"
if not defined SPECPMIN    set "SPECPMIN=0.6"
if not defined SPECNGEN    set "SPECNGEN=512"
if not defined SPECPORT    set "SPECPORT=8099"
if not defined SPECFIT     set "SPECFIT=-fit off"
if not defined SPECWAIT    set "SPECWAIT=900"
if not defined SPECTIMEOUT set "SPECTIMEOUT=1800"
if not defined SPECVERB    set "SPECVERB=4"
if not defined SPECREPGEN  set "SPECREPGEN=1024"
if not defined SPECREPLINE set "SPECREPLINE=80"
if not defined SPECARMS    set "SPECARMS=np1 np2-one np2-lock np2-free np1-noreuse ngram-off ngram-on ngram-crlf ngram-crlf-off ngram-miss"
if not defined SETTLE      set "SETTLE=30"

rem the two prompts. length does not matter here, only the generated tokens do, so the built-in
rem pair is short; point SPECPROMPTFILE1 and SPECPROMPTFILE2 at your own text to use that instead.
rem the second one only has to make the model write something else, otherwise np2-free would be
rem another np2-lock
if not defined SPECPROMPT1 set "SPECPROMPT1=Explain in detail how a modern operating system schedules threads on a multi core processor. Cover run queues, load balancing, priority inheritance and cache affinity."
if not defined SPECPROMPT2 set "SPECPROMPT2=Explain in detail how a relational database executes a join between two large tables. Cover hash join, merge join, nested loops, spilling to disk and cardinality estimation."

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

if /i "%SPECDRAFT%"=="%MODEL%" (
    echo SPECDRAFT is the target model
    echo it must be the MTP draft gguf, the one whose name starts with mtp-
    exit /b 1
)

rem a split gguf is loaded through its first shard. picking any other shard by hand makes
rem llama-server exit with "illegal split file idx" once the target is already loaded
if not "!SPECDRAFT:-of-=!"=="!SPECDRAFT!" if "!SPECDRAFT:-00001-of-=!"=="!SPECDRAFT!" (
    echo SPECDRAFT is not the first shard of a split gguf: %SPECDRAFT%
    echo the draft here is the MTP gguf, one file, not a shard of the target model
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

set "WORK=%LOGS%\17-spec-np-%TS%"
if not exist "%WORK%" mkdir "%WORK%"
set "SUM=%LOGS%\17-spec-np-%TS%-summary.txt"
set "RC=0"

echo ### 17-spec-np %TS% > "%SUM%"
echo ### MODEL=%MODEL% >> "%SUM%"
echo ### SPECDRAFT=%SPECDRAFT% >> "%SUM%"
echo ### n-max=%SPECNMAX% p-min=%SPECPMIN% ctx=%SPECCTX% n_predict=%SPECNGEN% >> "%SUM%"
echo ### repeat body: %SPECREPLINE% lines, n_predict=%SPECREPGEN% >> "%SUM%"
echo ### LOADMODE=%LOADMODE% EXTRA=%EXTRA% %SPECFIT% >> "%SUM%"
echo. >> "%SUM%"

call :mkreq one
call :mkreq lock
call :mkreq free
call :mkrep repeat      lf
call :mkrep repeat-crlf crlf
if not "%RC%"=="0" exit /b 1

set "ABORT=0"
for %%a in (%SPECARMS%) do call :arm %%a

set "LLAMA_GRAPH_REUSE_DISABLE="
set "LLAMA_TRACE="

echo.
echo summary in %SUM%
type "%SUM%"
exit /b %RC%

rem ------------------------------------------------------------------
rem %1 = body name: one = a single prompt, lock = the first prompt twice,
rem free = both prompts. powershell writes the body so the prompt does not have to survive
rem batch quoting, and WriteAllText keeps the utf8 BOM out of it
:mkreq
set "PMODE=%~1"
set "PFILE1=%SPECPROMPTFILE1%"
set "PFILE2=%SPECPROMPTFILE2%"
set "PTEXT1=%SPECPROMPT1%"
set "PTEXT2=%SPECPROMPT2%"
set "PN=%SPECNGEN%"
set "POUT=%WORK%\req-%PMODE%.json"
powershell -NoProfile -Command "$p1 = if ($env:PFILE1 -and (Test-Path -LiteralPath $env:PFILE1)) { Get-Content -Raw -LiteralPath $env:PFILE1 } else { $env:PTEXT1 }; $p2 = if ($env:PFILE2 -and (Test-Path -LiteralPath $env:PFILE2)) { Get-Content -Raw -LiteralPath $env:PFILE2 } else { $env:PTEXT2 }; if ($env:PMODE -eq 'one') { $p = $p1 } elseif ($env:PMODE -eq 'lock') { $p = @($p1, $p1) } else { $p = @($p1, $p2) }; $j = [ordered]@{ prompt = $p; n_predict = [int]$env:PN; temperature = 0; top_k = 1; ignore_eos = $true; cache_prompt = $false; stream = $false } | ConvertTo-Json -Compress -Depth 5; [System.IO.File]::WriteAllText($env:POUT, $j, (New-Object System.Text.UTF8Encoding($false)))"
if not exist "%POUT%" (
    echo failed to build the request body %POUT%
    set "RC=1"
)
goto :eof

rem ------------------------------------------------------------------
rem %1 = body name, %2 = line ending: lf or crlf. builds a block of text and asks the model to
rem write it back verbatim, which is the one shape ngram-mod is built for. the block is generated
rem here so the arm needs no data file; point SPECREPFILE at your own text to use that instead.
rem temperature 0 and ignore_eos keep every arm on the same token count, so the tail the model
rem writes after the block is the same in all of them and does not skew the comparison
:mkrep
set "PMODE=%~1"
set "PEOL=%~2"
set "PFILE=%SPECREPFILE%"
set "PLINES=%SPECREPLINE%"
set "PN=%SPECREPGEN%"
set "POUT=%WORK%\req-%PMODE%.json"
powershell -NoProfile -Command "$eol = if ($env:PEOL -eq 'crlf') { [string][char]13 + [string][char]10 } else { [string][char]10 }; if ($env:PFILE -and (Test-Path -LiteralPath $env:PFILE)) { $body = (Get-Content -Raw -LiteralPath $env:PFILE) -replace '\r\n', [string][char]10 } else { $body = [string]::Join([string][char]10, (1..[int]$env:PLINES | ForEach-Object { '    const int row_{0:d3} = accumulate(buffer, stride * {0}, offset + {0}, mask_{0:d3});' -f $_ })) }; $block = $body -replace [string][char]10, $eol; $p = 'Write the following block of text back to me, exactly as it is. Output only the block, no commentary and no code fences.' + $eol + $eol + $block; $j = [ordered]@{ prompt = $p; n_predict = [int]$env:PN; temperature = 0; top_k = 1; ignore_eos = $true; cache_prompt = $false; stream = $false } | ConvertTo-Json -Compress -Depth 5; [System.IO.File]::WriteAllText($env:POUT, $j, (New-Object System.Text.UTF8Encoding($false)))"
if not exist "%POUT%" (
    echo failed to build the request body %POUT%
    set "RC=1"
)
goto :eof

rem ------------------------------------------------------------------
rem %1 = arm name
:arm
rem one arm failing to start means the next four fail the same way, so stop instead
if "%ABORT%"=="1" goto :eof
set "ARM=%~1"
set "NP="
set "BODY="
set "NREQ="
set "STYPE=draft-mtp"
set "NGEN=%SPECNGEN%"
set "LLAMA_GRAPH_REUSE_DISABLE="
set "LLAMA_TRACE="

if "%ARM%"=="np1"         ( set "NP=1" & set "BODY=one"  & set "NREQ=1" )
if "%ARM%"=="np1-noreuse" ( set "NP=1" & set "BODY=one"  & set "NREQ=1" & set "LLAMA_GRAPH_REUSE_DISABLE=1" )
if "%ARM%"=="np2-one"     ( set "NP=2" & set "BODY=one"  & set "NREQ=1" )
if "%ARM%"=="np2-lock"    ( set "NP=2" & set "BODY=lock" & set "NREQ=2" )
if "%ARM%"=="np2-free"    ( set "NP=2" & set "BODY=free" & set "NREQ=2" )

rem LLAMA_TRACE=1 adds the per-step accept count and marks the steps that had to restore a
rem checkpoint, which is what makes these four readable. the np arms stay without it so their
rem logs still compare with the runs already written down in README.md
if "%ARM%"=="ngram-off"   ( set "NP=1" & set "BODY=repeat"      & set "NREQ=1" & set "NGEN=%SPECREPGEN%" & set "LLAMA_TRACE=1" )
if "%ARM%"=="ngram-on"    ( set "NP=1" & set "BODY=repeat"      & set "NREQ=1" & set "NGEN=%SPECREPGEN%" & set "LLAMA_TRACE=1" & set "STYPE=ngram-mod,draft-mtp" )
if "%ARM%"=="ngram-crlf"  ( set "NP=1" & set "BODY=repeat-crlf" & set "NREQ=1" & set "NGEN=%SPECREPGEN%" & set "LLAMA_TRACE=1" & set "STYPE=ngram-mod,draft-mtp" )
if "%ARM%"=="ngram-crlf-off" ( set "NP=1" & set "BODY=repeat-crlf" & set "NREQ=1" & set "NGEN=%SPECREPGEN%" & set "LLAMA_TRACE=1" )
if "%ARM%"=="ngram-miss"  ( set "NP=1" & set "BODY=one"         & set "NREQ=1" & set "LLAMA_TRACE=1" & set "STYPE=ngram-mod,draft-mtp" )

if not defined NP (
    echo unknown arm %ARM%, known: np1 np1-noreuse np2-one np2-lock np2-free ngram-off ngram-on ngram-crlf ngram-crlf-off ngram-miss
    set "RC=1"
    goto :eof
)

set "SRVLOG=%LOGS%\17-spec-np-%TS%-%ARM%.log"
set "SPECARGS=-m "%MODEL%" -md "%SPECDRAFT%" --spec-type "%STYPE%" --spec-draft-n-min 0 --spec-draft-n-max %SPECNMAX% --spec-draft-p-min %SPECPMIN% -c %SPECCTX% -np %NP% -fa on %SPECFIT% %LOADMODE% %EXTRA% --host 127.0.0.1 --port %SPECPORT% -lv %SPECVERB%"

echo === %ARM%: -np %NP%, body %BODY% (%NREQ% prompts), spec-type %STYPE%, graph_reuse_disable=%LLAMA_GRAPH_REUSE_DISABLE% -^> %SRVLOG%
set "SCMD=%WORK%\srv-%ARM%.cmd"
> "%SCMD%" echo @echo off
>>"%SCMD%" echo "%BIN%\llama-server.exe" %SPECARGS% ^> "%SRVLOG%" 2^>^&1
start "17-spec-np-%ARM%" /min cmd /c "%SCMD%"

call :waitup
if not "%UP%"=="1" (
    if "%GONE%"=="1" (
        echo llama-server.exe exited after %WAITED%s, last lines of %SRVLOG%:
    ) else (
        echo no 200 from /health after %WAITED%s, last lines of %SRVLOG%:
    )
    powershell -NoProfile -Command "Get-Content -Tail 20 -LiteralPath $env:SRVLOG"
    set "RC=1"
    set "ABORT=1"
    call :teardown
    goto :eof
)

for /f %%t in ('powershell -NoProfile -Command "(Get-Date).Ticks"') do set "TSTART=%%t"

curl.exe -s -S --max-time %SPECTIMEOUT% -H "Content-Type: application/json" --data-binary @"%WORK%\req-%BODY%.json" -o "%WORK%\%ARM%-resp.json" "http://127.0.0.1:%SPECPORT%/completion" 2> "%WORK%\%ARM%-curl.txt"
set "CURLRC=%ERRORLEVEL%"

set /a "TOTTOK=NREQ*NGEN"
rem both numbers come out of one call, formatted with InvariantCulture: a russian locale renders
rem Round() as 16,842 and the next command line then reads that as two arguments
for /f "tokens=1,2" %%x in ('powershell -NoProfile -Command "$c = [System.Globalization.CultureInfo]::InvariantCulture; $e = ((Get-Date).Ticks - %TSTART%) / 10000000.0; $r = %TOTTOK% / [math]::Max($e, 0.001); '{0} {1}' -f [math]::Round($e, 3).ToString($c), [math]::Round($r, 2).ToString($c)"') do (
    set "ELAPSED=%%x"
    set "AGGTPS=%%y"
)

if not "%CURLRC%"=="0" (
    echo curl failed with %CURLRC%, see %WORK%\%ARM%-curl.txt
    set "RC=1"
)

rem the slot prints its timings as it is released, give it a moment before the process goes away
powershell -NoProfile -Command "Start-Sleep -Seconds 2"
call :teardown

echo. >> "%SUM%"
echo ### %ARM%: -np %NP%, %NREQ% concurrent prompts, body %BODY%, spec-type %STYPE%, graph_reuse_disable=%LLAMA_GRAPH_REUSE_DISABLE% >> "%SUM%"
echo ### wall %ELAPSED%s for %TOTTOK% generated tokens = %AGGTPS% t/s aggregate, curl rc=%CURLRC% >> "%SUM%"
findstr /c:"eval time" /c:"graphs reused" /c:"draft acceptance" /c:"acc per pos" /c:"statistics " "%SRVLOG%" >> "%SUM%"
findstr /c:"n_rs_seq" /c:"recurrent" /c:"KV self size" /c:"compute buffer size" "%SRVLOG%" >> "%SUM%"
for /f %%c in ('findstr /c:"restore checkpoint" "%SRVLOG%" ^| find /c /v ""') do echo ### draft steps that restored a checkpoint: %%c >> "%SUM%"

echo === %ARM%: %AGGTPS% t/s aggregate over %ELAPSED%s

echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
goto :eof

rem ------------------------------------------------------------------
rem poll /health until it answers 200 or SPECWAIT runs out. the sleep comes first: the process
rem needs a moment to appear in tasklist, and checking before that would call the arm dead
:waitup
set "UP=0"
set "GONE=0"
set "WAITED=0"
:waitup_loop
powershell -NoProfile -Command "Start-Sleep -Seconds 3"
set /a "WAITED+=3"
set "CODE="
curl.exe -s -o nul -w "%%{http_code}" "http://127.0.0.1:%SPECPORT%/health" > "%WORK%\health.txt" 2>nul
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
if %WAITED% geq %SPECWAIT% goto :eof
goto :waitup_loop

rem ------------------------------------------------------------------
:teardown
taskkill /f /im llama-server.exe >nul 2>&1
goto :eof
