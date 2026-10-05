@echo off
setlocal enabledelayedexpansion
if /i "%~1"=="speed" goto :speed
call "%~dp0_config.bat"
where py >nul 2>&1
if not errorlevel 1 (
    py -3 "%~dp0_compare.py" memory-quality
) else (
    python "%~dp0_compare.py" memory-quality
)
if errorlevel 1 exit /b 1
call "%~dp023-spec-auto.bat" memory-quality
if errorlevel 1 exit /b 1
if /i "%~1"=="quality" exit /b 0
:speed
if not defined MEMCHARS if defined CUDAOVERLAPCHARS set "MEMCHARS=%CUDAOVERLAPCHARS%"
if not defined MEMCHARS set "MEMCHARS=16384 131072"
for %%n in (%MEMCHARS%) do (
    set "CUDAOVERLAPCHARS=%%n"
    call "%~dp023-spec-auto.bat" memory-speed
    if errorlevel 1 exit /b 1
)
exit /b 0
