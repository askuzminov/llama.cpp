@echo off
rem starts llama-server in router mode on a models preset: the daily launcher, not a measurement.
rem standalone - copy it anywhere, it needs no other file from this folder. run-all.bat skips it.
rem
rem how to stop it so the prompt cache reaches the disk:
rem   Ctrl+C, Ctrl+Break, closing the window, logging off and shutting the pc down all run the
rem   graceful path. The router asks every child instance to exit, the child hands its slots to the
rem   prompt cache and writes what is not on disk yet, and this script waits for all of it.
rem   taskkill /F does not: it kills the process outright and the live prompts are lost.
rem
rem two timeouts bound the wait:
rem   stop-timeout in models.ini (10 s by default) - after that the router force-kills the child
rem   HungAppTimeout in HKCU\Control Panel\Desktop (5 s by default) - after that windows kills us
rem     on a window close or a pc shutdown. Ctrl+C has no such limit
rem most of the time the wait is short anyway: while the slots are idle the server already copies
rem their states out to the spill dir, so the exit has little left to write
setlocal

rem git bash / msys puts its own find, sort and friends ahead of the windows ones on PATH
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%PATH%"

rem ---- edit these, or set them in the environment before calling this file ----

if not defined LLAMA_SERVER  set "LLAMA_SERVER=C:\Users\AI\Documents\github\llamacpp\llama.cpp\build-win\bin\Release\llama-server.exe"
if not defined MODELS_PRESET set "MODELS_PRESET=%USERPROFILE%\.config\llama-server\models.ini"
if not defined MODELS_MAX    set "MODELS_MAX=1"
if not defined SRVARGS       set "SRVARGS=--host 0.0.0.0 --port 12345"

rem where the models.ini entries point their cache-spill-dir. only read for the summary on exit
if not defined CACHEDIR      set "CACHEDIR=%USERPROFILE%\Downloads\llama-cache"

rem the console goes away with the window at shutdown, so keep a copy of the log next to this file.
rem this is where to look after a reboot to see whether the cache was written
if not defined LOGDIR        set "LOGDIR=%~dp0logs"

rem ---------------------------------------------------------------------------

if not exist "%LLAMA_SERVER%" (
    echo ERROR: llama-server.exe not found
    echo %LLAMA_SERVER%
    pause
    exit /b 1
)

if not exist "%MODELS_PRESET%" (
    echo ERROR: models.ini not found
    echo %MODELS_PRESET%
    pause
    exit /b 1
)

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%i"
set "SRVLOG=%LOGDIR%\serve-%TS%.log"

echo === llama-server --models-preset "%MODELS_PRESET%" --models-max %MODELS_MAX% %SRVARGS%
echo === log:         %SRVLOG%
echo === prompt cache %CACHEDIR%
echo.

"%LLAMA_SERVER%" ^
  --models-preset "%MODELS_PRESET%" ^
  --models-max %MODELS_MAX% ^
  --log-file "%SRVLOG%" ^
  --log-timestamps ^
  %SRVARGS%

set "RC=%ERRORLEVEL%"

rem Ctrl+C makes cmd ask "Terminate batch job (Y/N)?" and Y skips the rest of this file. the server
rem has already exited by then, so nothing is cut short - only this summary. the log file has it all
echo.
echo === llama-server exited with %RC%
powershell -NoProfile -Command "$f = @(Get-ChildItem -LiteralPath '%CACHEDIR%' -Filter state-*.bin -ErrorAction SilentlyContinue); '=== prompt cache: {0} files, {1:N1} MiB' -f $f.Count, ([double](($f | Measure-Object Length -Sum).Sum) / 1MB)"

if not "%RC%"=="0" pause
exit /b %RC%
