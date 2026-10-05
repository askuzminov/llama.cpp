@echo off
setlocal enabledelayedexpansion
call "%~dp0_config.bat"
if /i "%~1"=="speed" goto :speed
echo === 39-memory: KLD/PPL quality comparison
where py >nul 2>&1
if not errorlevel 1 (
    py -3 "%~dp0_compare.py" memory-quality
) else (
    python "%~dp0_compare.py" memory-quality
)
if errorlevel 1 (
    echo PP/TG not measured: the KLD/PPL quality stage failed. See memory-quality-*/summary.txt in "%LOGS%".
    exit /b 1
)
echo === 39-memory: greedy quality comparison
call "%~dp023-spec-auto.bat" memory-quality
if errorlevel 1 (
    echo PP/TG not measured: the greedy quality stage failed. See the 23-spec-auto summary in "%LOGS%".
    exit /b 1
)
if /i "%~1"=="quality" exit /b 0
:speed
echo === 39-memory: PP/TG speed comparison
if not defined MEMCHARS if defined CUDAOVERLAPCHARS set "MEMCHARS=%CUDAOVERLAPCHARS%"
if not defined MEMCHARS set "MEMCHARS=16384 131072"
for %%n in (%MEMCHARS%) do (
    set "CUDAOVERLAPCHARS=%%n"
    call "%~dp023-spec-auto.bat" memory-speed
    if errorlevel 1 exit /b 1
)
exit /b 0
