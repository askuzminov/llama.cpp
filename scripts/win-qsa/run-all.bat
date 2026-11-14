@echo off
rem the build, the upstream build of 07 and every step with the variants of the open questions, in order: 00-build,
rem 00-build upstream, 04-micro (off), 01-check, 06-decode, 03-profile, 07-kld, 07-kld against the exact f32 base
rem (the 395 only), 02-long. about 3 hours on the 395, more on the 3090 (two CUDA builds, more variants; README).
rem each step runs only what the same sources have not measured yet, so after a change of the code everything runs
rem and after a change of the variants only the new ones. the variants below are set only where _local.bat has none;
rem RUN_BUILD, RUN_UPSTREAM, RUN_MICRO, RUN_CHECK, RUN_DECODE, RUN_PROFILE, RUN_KLD, RUN_EXACT, RUN_LONG=0 there
rem leave a step out. serve.bat is not part of it and has to be stopped
setlocal enabledelayedexpansion
rem the folder of this file, for the subroutine below: there the zeroth argument is the label, not this file
set "HERE=%~dp0"
call "%HERE%_config.bat"

rem the questions of 14.11 (README, run-all). the 395: device memory against other processes (13.11 a reconnected
rem remote desktop took its memory back, Windows moved part of the server to the shared memory, the system swapped).
rem 14.11 15:25-15:50 (f348322df): 01 green on both machines; 03 default without trims, idle1 (LLAMA_KV_TRIM_IDLE_S=1)
rem gave back 4968 MiB in two idle gaps, evict600 (LLAMA_KV_TEST_DEMOTED_MIB=600) moved slots to the prompt cache and
rem the next requests restored them (38062 tokens: 864 MiB in 92 ms, restored in ~0.25 s), prompt and verify times the
rem same in all three. two of its three moves were slots within the step of the elastic KV (13 and 4014 tokens, 0 MiB
rem given back): now only slots over the step move. 03 evict600 once more (one move expected, the long slot); 01 off
rem (the change is in the idle path of the server only); the 3090 has nothing to run. the reconnect itself is a test
rem by hand with serve.bat (README: the lines "the system moved about ... MiB" and "moved to the prompt cache"). settled and left out: on the 395 mm and radix0 (01.11, 02.11), the 07 split of 12.11 (copy,
rem dense, nofuse, radix0); on the 3090 nowin (02.11), and 12.11 cap7 (-1.1%), full (the whole vocabulary of the MTP
rem head, -0.3%), m256, upmoe0 (-1.6%), upmoe3, radix-up in 03 (the prompt within 1.5%), m512 (the card ran full and
rem the guard refused 256 bytes: 512 MiB next to the expert cache is too little); 13.11 upmoe5/upmoe8 (the cache of
rem upstream -20%/-10% TG in two runs) and the 07 split (copy = upstream bit for bit, the CPU: fork 1.9434 against
rem upstream 1.9442); again (a second default) stays: the logits of default bit for bit
if not defined QA_PAR_ARMS set "QA_PAR_ARMS=default;rows0 LLAMA_QSA_ROWS=0"
rem the probes of 02-long take the default only, the variants of the probes are a question for 01-check
if not defined LONG_QARMS set "LONG_QARMS=default"
rem the indexer kernels have not changed since the micro of 31.10 evening: 04-micro only on request (RUN_MICRO=1)
if not defined RUN_MICRO set "RUN_MICRO=0"
if /i "%BACKEND%"=="cuda" goto :arms_cuda
if not defined QA_QARMS set "QA_QARMS=default"
if not defined QA_SPEC_ARMS set "QA_SPEC_ARMS=default"
rem 03 at three depths: one trim and growth after each of the first two in idle1; evict600: the estimate of moved
rem memory forced to 600 MiB (LLAMA_KV_TEST_DEMOTED_MIB), so that the first idle second with a slot moves it to the
rem prompt cache and the next request restores it from there (once: the given back memory covers the 600 MiB)
if not defined QA_PARMS set "QA_PARMS=evict600 LLAMA_KV_TEST_DEMOTED_MIB=600"
if not defined RUN_CHECK set "RUN_CHECK=0"
if not defined QA_PROF_DEPTHS set "QA_PROF_DEPTHS=0 65536 131072"
if not defined QA_DARMS set "QA_DARMS=default"
if not defined RUN_DECODE set "RUN_DECODE=0"
if not defined QA_KARMS set "QA_KARMS=fork;upstream @upstream"
rem decode4 and decode8: the default at ubatch 4 and 8 (the draft checks; 4 on the two planes, 8 on the tile), the
rem arms of 12.11 (0.0311 and 0.0307 then): their column before
if not defined EXACT_KARMS set "EXACT_KARMS=decode4 QA_ARM_UB=4;decode8 QA_ARM_UB=8"
if not defined RUN_EXACT set "RUN_EXACT=0"
if not defined RUN_KLD set "RUN_KLD=0"
if not defined RUN_LONG set "RUN_LONG=0"
goto :arms_done
:arms_cuda
if not defined QA_QARMS set "QA_QARMS=default;again LLAMA_QA_AGAIN=1"
if not defined QA_PARMS set "QA_PARMS=default"
if not defined QA_DARMS set "QA_DARMS=default"
rem nothing to run on the 3090 this time (the change is in the idle path of the server, which gives nothing back here)
if not defined RUN_CHECK set "RUN_CHECK=0"
if not defined QA_KARMS set "QA_KARMS=fork;upstream @upstream"
if not defined RUN_DECODE set "RUN_DECODE=0"
if not defined RUN_PROFILE set "RUN_PROFILE=0"
if not defined RUN_KLD set "RUN_KLD=0"
if not defined RUN_LONG set "RUN_LONG=0"
rem the CUDA matmuls have no f32 mode: 07 against the exact base runs on the 395 only
if not defined RUN_EXACT set "RUN_EXACT=0"
:arms_done
rem the decode tasks with a long prompt (about 24K tokens) and a short reply: the prompt speed of each variant too
if not defined QA_DEC_SET set "QA_DEC_SET=prose ru code-py code-cpp edit json long"
rem 05-mm (the dense q8_0 matmuls by tile) only where a section above or _local.bat asks (RUN_MM=1)
if not defined RUN_MM set "RUN_MM=0"

for %%v in (BUILD MICRO CHECK DECODE PROFILE KLD EXACT LONG) do if not defined RUN_%%v set "RUN_%%v=1"
rem the upstream build is the base of 07 against upstream
if not defined RUN_UPSTREAM set "RUN_UPSTREAM=%RUN_KLD%"
set "STEPS="
if "%RUN_BUILD%"=="1"    set "STEPS=!STEPS! 00-build"
if "%RUN_UPSTREAM%"=="1" set "STEPS=!STEPS! 00-upstream"
if "%RUN_MICRO%"=="1"    set "STEPS=!STEPS! 04-micro"
if "%RUN_MM%"=="1"       set "STEPS=!STEPS! 05-mm"
if "%RUN_CHECK%"=="1"    set "STEPS=!STEPS! 01-check"
if "%RUN_DECODE%"=="1"   set "STEPS=!STEPS! 06-decode"
if "%RUN_PROFILE%"=="1"  set "STEPS=!STEPS! 03-profile"
if "%RUN_KLD%"=="1"      set "STEPS=!STEPS! 07-kld"
if "%RUN_EXACT%"=="1"    set "STEPS=!STEPS! 07-exact"
if "%RUN_LONG%"=="1"     set "STEPS=!STEPS! 02-long"
if "%STEPS%"=="" (
    echo every RUN_ step is 0, nothing to do
    exit /b 1
)

rem the probes of 01 and 02, the decode tasks and 07 read a long text
if not defined QA_CORPUS if not exist "%PPLFILE%" call "%HERE%get-wikitext.bat"

echo steps:%STEPS%
echo variants: micro [%QA_MARMS%] check [%QA_QARMS%] decode [%QA_DARMS%] profile [%QA_PARMS%] kld [%QA_KARMS%]
echo   exact [%EXACT_KARMS%] long [%LONG_QARMS%] par [%QA_PAR_ARMS%], upstream %UPSTREAMREF%
rem 00-build waits for a key after a failure, run-all goes on by itself
set "QSA_UNATTENDED=1"
rem not RC: cmake takes RC from the environment as the resource compiler of a fresh configure
set "STEPRC=0"
set "RESULTS="
set "CHECK_QARMS=%QA_QARMS%"
set "KLD_KARMS=%QA_KARMS%"
for %%s in (%STEPS%) do (
    echo === %%s
    call :step %%s
    set "EC=!ERRORLEVEL!"
    if not "!EC!"=="0" set "STEPRC=!EC!"
    set "RESULTS=!RESULTS! %%s=!EC!"
    echo.
    rem the steps after a failed build would measure the old binaries
    if "%%s"=="00-build" if not "!EC!"=="0" goto :done
)

:done
echo === exit codes:%RESULTS%
echo the summary of each step is in its folder in %LOGS%
exit /b %STEPRC%

rem one step with its variants: 02-long reads the probes of LONG_QARMS, 07-exact compares with the exact f32 base. a
rem failed 00-upstream leaves 07 against upstream to its check of the build (UPSTREAMREF), the other steps go on
:step
set "QA_QARMS=%CHECK_QARMS%"
set "QA_KARMS=%KLD_KARMS%"
set "QA_KLD_BASE=upstream"
if "%~1"=="00-upstream" goto :step_upstream
if "%~1"=="07-exact" goto :step_exact
if "%~1"=="02-long" set "QA_QARMS=%LONG_QARMS%"
call "%HERE%%~1.bat"
exit /b %ERRORLEVEL%
:step_upstream
call "%HERE%00-build.bat" upstream
exit /b %ERRORLEVEL%
:step_exact
set "QA_KLD_BASE=exact"
set "QA_KARMS=%EXACT_KARMS%"
call "%HERE%07-kld.bat"
exit /b %ERRORLEVEL%
