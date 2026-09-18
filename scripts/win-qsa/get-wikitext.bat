@echo off
rem downloads the text used by 05 and 06 next to these scripts. a zip already
rem lying here is used as is, so it can be copied by hand from another machine.
rem PPLURL in _local.bat points at a different source.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

set "DIR=%~dp0"
set "ZIP=%DIR%wikitext-2-raw-v1.zip"
if not defined PPLURL set "PPLURL=https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

if exist "%PPLFILE%" (
    echo already there
    exit /b 0
)

if exist "%ZIP%" goto :unpack

echo downloading %PPLURL%
where curl.exe >nul 2>&1
if not "%ERRORLEVEL%"=="0" (
    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PPLURL%' -OutFile '%ZIP%'"
) else (
    curl.exe -L --fail --retry 2 -o "%ZIP%" "%PPLURL%"
)
set "EC=!ERRORLEVEL!"
if not "!EC!"=="0" if exist "%ZIP%" del "%ZIP%"

if not exist "%ZIP%" (
    echo.
    echo download failed, exit=!EC!
    echo 05 and 06 need %PPLFILE%
    echo put wikitext-2-raw-v1.zip next to these scripts and run this again,
    echo or set PPLURL in _local.bat to a source this machine can reach
    exit /b 1
)

:unpack
powershell -NoProfile -Command "Expand-Archive -Force -Path '%ZIP%' -DestinationPath '%DIR%'"
if not "%ERRORLEVEL%"=="0" exit /b 1
del "%ZIP%"

if not exist "%PPLFILE%" (
    echo unpacked, but %PPLFILE% is still missing
    exit /b 1
)

dir /b "%DIR%wikitext-2-raw"
