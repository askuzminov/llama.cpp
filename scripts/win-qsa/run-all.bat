@echo off
rem every step of this folder, in order, one log each. 14-server is the only one left out:
rem it runs until stopped. which steps run is set by RUN_* in _config.bat, all of them are
rem 1 by default; the exit code of each is collected into one summary at the end.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem defaults, in case _config.bat on this machine predates the RUN_* switches
if not defined RUN_BUILD   set "RUN_BUILD=1"
if not defined RUN_PREFILL set "RUN_PREFILL=1"
if not defined RUN_QUALITY set "RUN_QUALITY=1"
if not defined RUN_KL      set "RUN_KL=1"
if not defined RUN_TIMING  set "RUN_TIMING=1"
if not defined RUN_DIAG    set "RUN_DIAG=1"
if not defined RUN_PLE     set "RUN_PLE=1"
if not defined RUN_DISK    set "RUN_DISK=1"
if not defined RUN_MMID    set "RUN_MMID=1"

set "STEPS="
if "%RUN_BUILD%"=="1"   set "STEPS=%STEPS% 00-build"
if "%RUN_PREFILL%"=="1" set "STEPS=%STEPS% 01-fa-correct 02-fa-perf 03-prefill 04-fa-sparse"
if "%RUN_QUALITY%"=="1" set "STEPS=%STEPS% 05-perplexity"
if "%RUN_KL%"=="1"      set "STEPS=%STEPS% 06-kl-divergence"
if "%RUN_TIMING%"=="1"  set "STEPS=%STEPS% 07-perf-logger"
if "%RUN_DIAG%"=="1"    set "STEPS=%STEPS% 08-model-check"
if "%RUN_TIMING%"=="1"  set "STEPS=%STEPS% 09-alloc-timing 10-load-time"
if "%RUN_PLE%"=="1"     set "STEPS=%STEPS% 11-ple-cache 12-ple-real"
if "%RUN_DISK%"=="1"    set "STEPS=%STEPS% 13-disk-iops"
if "%RUN_MMID%"=="1"    set "STEPS=%STEPS% 15-mmid"

if "%STEPS%"=="" (
    echo every RUN_* is 0, nothing to do
    exit /b 1
)

rem 05, 06 and 12 need the wikitext text, drop them if it cannot be fetched
if "%RUN_QUALITY%%RUN_KL%%RUN_PLE%"=="000" goto :no_text
if not exist "%PPLFILE%" call "%~dp0get-wikitext.bat"
if not exist "%PPLFILE%" (
    echo no wikitext text, skipping 05, 06 and 12
    set "STEPS=!STEPS: 05-perplexity=!"
    set "STEPS=!STEPS: 06-kl-divergence=!"
    set "STEPS=!STEPS: 12-ple-real=!"
)
:no_text

echo steps:!STEPS!
echo.

rem 06 asks before writing its ~10 GB logits file, run-all has already decided
set "QSA_UNATTENDED=1"

set "SUM=%LOGS%\run-all-%TS%-summary.txt"
echo ### run-all %TS% > "%SUM%"
echo ### MODEL=%MODEL% >> "%SUM%"
echo ### LOADMODE=%LOADMODE% EXTRA=%EXTRA% >> "%SUM%"
echo. >> "%SUM%"

set "RC=0"

for %%s in (%STEPS%) do (
    echo === %%s
    call "%~dp0%%s.bat"
    set "EC=!ERRORLEVEL!"
    if not "!EC!"=="0" set "RC=!EC!"
    echo %%s exit=!EC! >> "%SUM%"
    echo === %%s exit=!EC!
    echo.

    rem a failed build makes every step after it measure the old binaries, or nothing
    if "%%s"=="00-build" if not "!EC!"=="0" (
        echo build failed, stopping
        goto :done
    )
)

:done
echo.
echo === summary
type "%SUM%"
echo.
echo logs are in %LOGS%
exit /b %RC%
