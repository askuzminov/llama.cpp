@echo off
rem the build and every measurement, in order: 00-build, 01-check, 02-bench, 03-quality, 04-decode, 05-profile.
rem each one runs only what the same sources have not measured yet, so after a change of the code everything
rem runs, and after a change of the scripts only the new arms do. RUN_* in _local.bat turn steps off (05 is on
rem for the 395 only); serve.bat is not part of it.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

set "STEPS="
if "%RUN_BUILD%"=="1"   set "STEPS=%STEPS% 00-build"
if "%RUN_CHECK%"=="1"   set "STEPS=%STEPS% 01-check"
if "%RUN_BENCH%"=="1"   set "STEPS=%STEPS% 02-bench"
if "%RUN_QUALITY%"=="1" set "STEPS=%STEPS% 03-quality"
if "%RUN_DECODE%"=="1"  set "STEPS=%STEPS% 04-decode"
if "%RUN_PROFILE%"=="1" set "STEPS=%STEPS% 05-profile"
if "%STEPS%"=="" (
    echo every RUN_* is 0, nothing to do
    exit /b 1
)

if "%RUN_QUALITY%"=="1" if not exist "%PPLFILE%" call "%~dp0get-wikitext.bat"

rem 00-build waits for a key after a failure, run-all goes on by itself
set "QSA_UNATTENDED=1"
set "RC=0"
set "RESULTS="
for %%s in (%STEPS%) do (
    echo === %%s
    call "%~dp0%%s.bat"
    set "EC=!ERRORLEVEL!"
    if not "!EC!"=="0" set "RC=!EC!"
    set "RESULTS=!RESULTS! %%s=!EC!"
    echo.
    rem the steps after a failed build would measure the old binaries
    if "%%s"=="00-build" if not "!EC!"=="0" goto :done
)

:done
echo === exit codes:%RESULTS%
echo the summary of each step is in its folder in %LOGS%
exit /b %RC%
