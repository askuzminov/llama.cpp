@echo off
rem куда уходит время выделения буферов vulkan: createBuffer, allocateMemory или
rem mapMemory. заодно перебирает размер блока, на которые режется контекст.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem обе ручки vulkan-овские, на другом бэкенде это четыре одинаковых загрузки
if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 09-alloc-timing is vulkan only
    exit /b 0
)

if not exist "%BIN%\llama-bench.exe" (
    echo not built: %BIN%\llama-bench.exe
    echo run 00-build.bat first
    exit /b 1
)

if not exist "%MODEL%" (
    echo model not found: %MODEL%
    echo set MODEL in _local.bat
    exit /b 1
)

set "LOG=%LOGS%\09-alloc-timing-%TS%.log"
echo writing %LOG%

echo ### MODEL=%MODEL% > "%LOG%"
echo ### LOADMODE=%LOADMODE% >> "%LOG%"

set "GGML_VK_ALLOC_TIMING=1"
set "RC=0"

rem 0 = не задавать переменную, взять умолчание бэкенда (1 GiB)
for %%b in (0 268435456 1073741824 4294967296) do (
    echo === block=%%b
    echo. >> "%LOG%"
    echo ### GGML_VK_SUBALLOCATION_BLOCK_SIZE=%%b >> "%LOG%"
    if "%%b"=="0" (set "GGML_VK_SUBALLOCATION_BLOCK_SIZE=") else (set "GGML_VK_SUBALLOCATION_BLOCK_SIZE=%%b")
    "%BIN%\llama-bench.exe" -m "%MODEL%" -v -fa on -p 8 -n 0 -ub 512 -r 1 --no-warmup %LOADMODE% %EXTRA% >> "%LOG%" 2>&1
    set "EC=!ERRORLEVEL!"
    if not "!EC!"=="0" set "RC=!EC!"
    echo ### exit=!EC! >> "%LOG%"
)
set "GGML_VK_ALLOC_TIMING="
set "GGML_VK_SUBALLOCATION_BLOCK_SIZE="

echo.
echo done, %LOG%
exit /b %RC%
