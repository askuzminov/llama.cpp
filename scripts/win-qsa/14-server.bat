@echo off
rem starts llama-server with the settings of this folder: the same model, load mode and
rem backend arguments as the measurements, but the full context the model was trained with.
rem this one is not a measurement and run-all.bat does not call it - it runs until stopped.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem context the server serves. the research scripts stay shallow on purpose, a deep sweep
rem costs hours and the trend is visible without it; the server has no such reason, so it
rem runs at the trained context of qwen4exp. 0 takes it from the model instead
if not defined SRVCTX  set "SRVCTX=262144"
if not defined SRVARGS set "SRVARGS=--host 127.0.0.1 --port 8080"

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

rem a sweep leaves this set in the shell it ran in, the server always runs the default
set "GGML_VK_FA_SPARSE="

echo === llama-server -c %SRVCTX% %LOADMODE% %EXTRA% %SRVARGS%
"%BIN%\llama-server.exe" -m "%MODEL%" -c %SRVCTX% -fa on %LOADMODE% %EXTRA% %SRVARGS%
exit /b %ERRORLEVEL%
