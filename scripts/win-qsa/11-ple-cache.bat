@echo off
rem кеш строк таблицы PLE (-lzm dio) против ленивого mmap (-lzm on). таблица PLE
rem у qwen4exp занимает около 27 ГБ: с -lzm on она остаётся в page cache и читается
rem случайными хардфолтами по 4 КиБ, с -lzm dio её строки собирает свой блочный кеш
rem с фиксированным бюджетом. в логе смотреть:
rem   "stays on disk"       - таблица не грузится, размер кеша, читатели, длина запроса
rem   "block lookups hit"   - доля попаданий, сколько запросов и какого размера
rem   "in flight"           - реальная глубина очереди: время внутри читателей, делённое
rem                           на время ожидания. ~1 при многих читателях значит, что
rem                           запросы где-то сериализуются, а не что диск медленный
rem   "per request"         - задержка одного запроса и полученная полоса
rem   "distinct blocks"     - рабочее множество реального прогона
rem   таблицу llama-bench   - что это дало префилу на каждой глубине
rem нужен -v, иначе llama-bench глушит лог загрузчика.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem каждый вариант это "читатели блокБайт блоковНаЗапрос", 0 = авто/по умолчанию
if not defined PLEVARIANTS set "PLEVARIANTS="1 0 64" "4 0 64" "16 0 64" "32 0 64""
if not defined REPS set "REPS=1"

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

set "LOG=%LOGS%\11-ple-cache-%TS%.log"
echo writing %LOG%

echo ### MODEL=%MODEL% > "%LOG%"
echo ### EXTRA=%EXTRA% >> "%LOG%"
echo ### DEPTHS=%DEPTHS% NPROMPT=%NPROMPT% REPS=%REPS% >> "%LOG%"
echo ### PLEVARIANTS=%PLEVARIANTS% >> "%LOG%"

set "LLAMA_ROW_CACHE_STATS=1"
if not defined SETTLE set "SETTLE=30"
set "RC=0"
set "FIRST=1"

rem базовая линия: ленивое чтение через mmap, настройки кеша ни при чём
call :run on 0 0 0

for %%v in (%PLEVARIANTS%) do (
    for /f "tokens=1-3" %%a in (%%v) do call :run dio %%a %%b %%c
)

set "LLAMA_ROW_CACHE_STATS="
set "LLAMA_ROW_CACHE_THREADS="
set "LLAMA_ROW_CACHE_BLOCK="
set "LLAMA_ROW_CACHE_RUN="

echo.
echo === summary
echo. >> "%LOG%"
echo ### summary >> "%LOG%"
set "PICK=/C:"### -lzm" /C:"stays on disk" /C:"block lookups hit" /C:"in flight" /C:"per request" /C:"distinct blocks" /C:"pp2048" /C:"lazy read enabled" /C:"error loading model" /C:"### exit=""
findstr %PICK% "%LOG%"
findstr %PICK% "%LOG%" >> "%LOG%"

echo.
echo done, %LOG%
exit /b %RC%

rem %1 = значение -lzm, дальше читатели, блок байт, блоков на запрос; 0 = по умолчанию
:run
if "%~2"=="0" (set "LLAMA_ROW_CACHE_THREADS=") else (set "LLAMA_ROW_CACHE_THREADS=%~2")
if "%~3"=="0" (set "LLAMA_ROW_CACHE_BLOCK=") else (set "LLAMA_ROW_CACHE_BLOCK=%~3")
if "%~4"=="0" (set "LLAMA_ROW_CACHE_RUN=") else (set "LLAMA_ROW_CACHE_RUN=%~4")
call :settle
echo === -lzm %~1 readers=%~2 block=%~3 B run=%~4
echo. >> "%LOG%"
echo ### -lzm %~1 readers=%~2 blockB=%~3 run=%~4 >> "%LOG%"
"%BIN%\llama-bench.exe" -m "%MODEL%" -v -fa on -p %NPROMPT% -n 0 -b 4096 -ub 512 -d %DEPTHS% -r %REPS% -lm dio -lzm %~1 %EXTRA% --progress -o md >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo ### exit=%EC% >> "%LOG%"
goto :eof

rem the gpu is freed lazily, a run started right after the previous one dies in
rem vkAllocateMemory. the first run has nothing to wait for
:settle
if "%FIRST%"=="1" (
    set "FIRST=0"
    goto :eof
)
echo === waiting %SETTLE%s for the gpu to be released
powershell -NoProfile -Command "Start-Sleep -Seconds %SETTLE%"
goto :eof
