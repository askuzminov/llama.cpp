@echo off
rem то же сравнение -lzm on / -lzm dio, но на настоящем тексте, а не на случайных
rem токенах llama-bench. это важно: кеш строк живёт за счёт того, что частые токены
rem повторяются, а llama-bench подаёт равномерно случайные id по всему словарю и
rem повторов там почти нет - для кеша это худший из возможных случаев.
rem llama-perplexity прогоняет префил по чанкам реального текста, так что тут видно
rem и скорость, и качество: PPL у on и dio обязан совпасть, иначе кеш врёт.
rem в логе смотреть:
rem   "Final estimate"      - должно совпадать между режимами
rem   "seconds per pass"    - скорость префила на реальных токенах
rem   "block lookups hit"   - доля попаданий на реальном распределении
rem   "read path:"         - развёртка пути чтения, если задан LLAMA_ROW_CACHE_BENCH
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not defined PLECHUNKS  set "PLECHUNKS=16"

if not exist "%BIN%\llama-perplexity.exe" (
    echo not built: %BIN%\llama-perplexity.exe
    echo run 00-build.bat first
    exit /b 1
)

if not exist "%MODEL%" (
    echo model not found: %MODEL%
    echo set MODEL in _local.bat
    exit /b 1
)

if not exist "%PPLFILE%" call "%~dp0get-wikitext.bat"
if not exist "%PPLFILE%" (
    echo missing %PPLFILE%, run get-wikitext.bat first
    exit /b 1
)

set "LOG=%LOGS%\12-ple-real-%TS%.log"
echo writing %LOG%

echo ### MODEL=%MODEL% > "%LOG%"
echo ### EXTRA=%EXTRA% >> "%LOG%"
echo ### PPLFILE=%PPLFILE% CHUNKS=%PLECHUNKS% >> "%LOG%"

set "RC=0"

call :run on
call :run dio

echo.
echo === summary
echo. >> "%LOG%"
echo ### summary >> "%LOG%"
set "PICK=/C:"### -lzm" /C:"Final estimate" /C:"seconds per pass" /C:"stays on disk" /C:"read path:" /C:"block lookups hit" /C:"in flight" /C:"per request" /C:"### exit=""
findstr %PICK% "%LOG%"
findstr %PICK% "%LOG%" >> "%LOG%"

echo.
echo done, %LOG%
exit /b %RC%

rem %1 = значение -lzm
:run
echo === -lzm %~1
echo. >> "%LOG%"
echo ### -lzm %~1 >> "%LOG%"
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c 8192 --chunks %PLECHUNKS% -fa on -lm dio -lzm %~1 %EXTRA% -v >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo ### exit=%EC% >> "%LOG%"
goto :eof
