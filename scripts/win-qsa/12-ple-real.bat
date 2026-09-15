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
rem   "distinct blocks"     - рабочее множество реального текста
rem   "read path:"         - развёртка пути чтения, если задан LLAMA_ROW_CACHE_BENCH
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem варианты кеша строк: "бюджетМиБ блокБайт", 0 = по умолчанию, по одному прогону на каждый.
rem осталось два вопроса: стоит ли пул чего-нибудь поверх файлового кеша системы и что даёт
rem блок больше строки. блок 4 КиБ читал 4.47 ГиБ там, где сами строки занимают 0.25 ГиБ
if not defined PLEREALVARIANTS set "PLEREALVARIANTS="0 0" "8 0" "1024 0""
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
echo ### PLEREALVARIANTS=%PLEREALVARIANTS% >> "%LOG%"

set "LLAMA_ROW_CACHE_STATS=1"
set "RC=0"

call :run on 0 0

for %%v in (%PLEREALVARIANTS%) do for /f "tokens=1-2" %%a in (%%v) do call :run dio %%a %%b

set "LLAMA_ROW_CACHE_STATS="
set "LLAMA_ROW_CACHE_MIB="
set "LLAMA_ROW_CACHE_BLOCK="

echo.
echo === summary
echo. >> "%LOG%"
echo ### summary >> "%LOG%"
set "PICK=/C:"### -lzm" /C:"Final estimate" /C:"seconds per pass" /C:"stays on disk" /C:"read path:" /C:"block lookups hit" /C:"in flight" /C:"per request" /C:"distinct blocks" /C:"### exit=""
findstr %PICK% "%LOG%"
findstr %PICK% "%LOG%" >> "%LOG%"

echo.
echo done, %LOG%
exit /b %RC%

rem %1 = значение -lzm, %2 = бюджет кеша в МиБ (0 = по умолчанию), %3 = размер блока в байтах
:run
if "%~2"=="0" (set "LLAMA_ROW_CACHE_MIB=") else (set "LLAMA_ROW_CACHE_MIB=%~2")
if "%~3"=="0" (set "LLAMA_ROW_CACHE_BLOCK=") else (set "LLAMA_ROW_CACHE_BLOCK=%~3")
echo === -lzm %~1 cache=%~2 MiB block=%~3
echo. >> "%LOG%"
echo ### -lzm %~1 LLAMA_ROW_CACHE_MIB=%~2 LLAMA_ROW_CACHE_BLOCK=%~3 >> "%LOG%"
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c 8192 --chunks %PLECHUNKS% -fa on -lm dio -lzm %~1 %EXTRA% -v >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo ### exit=%EC% >> "%LOG%"
goto :eof
