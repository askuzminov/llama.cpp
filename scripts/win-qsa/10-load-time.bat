@echo off
rem сколько занимает загрузка модели, по паре "-lm -lzm" из LOADVARIANTS.
rem выделение буферов идет параллельно с чтением файла, поэтому в логе интересны:
rem   "### wall="                - полное время прогона, от запуска до выхода
rem   "buffer allocation took"   - время в vkAllocateMemory/mapMemory
rem   "load_all_data: read"      - сколько и с какой скоростью прочитано
rem   "lazy tensors are mapped"  - файл отображен, и unbuffered чтения того же файла
rem                                идут через согласование с page cache, по одному
rem если чтение длиннее выделения, значит выделение спрятано целиком.
rem нужен -v: без него загрузчик молчит. "load time" из common_perf_print НЕ
rem показывает загрузку модели, это таймер контекста, на него не смотреть.
rem порядок вариантов важен: сначала те, что файл не отображают и не кладут его в
rem page cache. у none и mmap второй прогон читает файл уже из page cache и потому
rem быстрее первого - сравнивать между вариантами надо первые прогоны.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not exist "%BIN%\llama-bench.exe" (
    echo not built: %BIN%\llama-bench.exe
    echo run 00-build.bat first
    exit /b 1
)

rem вариант это "loadMode lazyMode", по одной загрузке модели на прогон
if not defined LOADVARIANTS set "LOADVARIANTS="dio dio" "dio off" "dio auto" "dio on" "none on" "mmap on""
if not defined LOADREPS    set "LOADREPS=2"

set "LOG=%LOGS%\10-load-time-%TS%.log"
echo writing %LOG%

echo ### MODEL=%MODEL% > "%LOG%"
echo ### EXTRA=%EXTRA% >> "%LOG%"
echo ### LOADVARIANTS=%LOADVARIANTS% LOADREPS=%LOADREPS% >> "%LOG%"

if not defined SETTLE set "SETTLE=30"

set "RC=0"
set "FIRST=1"

for %%v in (%LOADVARIANTS%) do for /f "tokens=1,2" %%a in (%%v) do (
    for /l %%r in (1,1,%LOADREPS%) do call :run %%a %%b %%r
)

echo.
echo === summary
echo. >> "%LOG%"
echo ### summary >> "%LOG%"
set "PICK=/C:"### -lm" /C:"load_mode =" /C:"lazy tensors are mapped" /C:"stays on disk" /C:"buffer allocation took" /C:"load_all_data: read" /C:"error loading model" /C:"### wall=""
findstr %PICK% "%LOG%"
findstr %PICK% "%LOG%" >> "%LOG%"

echo.
echo done, %LOG%
exit /b %RC%

rem %1 = -lm, %2 = -lzm, %3 = номер прогона
:run
call :settle
echo === -lm %~1 -lzm %~2 run %~3
echo. >> "%LOG%"
echo ### -lm %~1 -lzm %~2 run %~3 >> "%LOG%"
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).Ticks"') do set "T0=%%i"
"%BIN%\llama-bench.exe" -m "%MODEL%" -v -fa on -p 8 -n 0 -ub 512 -r 1 --no-warmup -lm %~1 -lzm %~2 %EXTRA% >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
for /f %%i in ('powershell -NoProfile -Command "[int](((Get-Date).Ticks-%T0%)/1e7)"') do set "WALL=%%i"
if not "%EC%"=="0" set "RC=%EC%"
echo ### wall=%WALL%s exit=%EC% >> "%LOG%"
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
