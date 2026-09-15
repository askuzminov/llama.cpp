@echo off
rem why the model does not load. llama-bench silences the loader log unless -v is
rem given, so this runs the smallest possible bench with -v and walks the load
rem options one at a time.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not exist "%BIN%\llama-bench.exe" (
    echo not built: %BIN%\llama-bench.exe
    echo run 00-build.bat first
    exit /b 1
)

set "LOG=%LOGS%\08-model-check-%TS%.log"
echo writing %LOG%

echo ### MODEL=%MODEL% > "%LOG%"
if exist "%MODEL%" (echo ### file exists >> "%LOG%") else (echo ### FILE DOES NOT EXIST >> "%LOG%")
echo. >> "%LOG%"
echo ### dir of the model folder >> "%LOG%"
for %%f in ("%MODEL%") do dir "%%~dpf" >> "%LOG%" 2>&1
echo. >> "%LOG%"

set "RC=0"

for %%c in ("dio on" "dio dio" "none on" "auto on") do (
    for /f "tokens=1,2" %%a in (%%c) do (
        echo === -lm %%a -lzm %%b
        echo. >> "%LOG%"
        echo ### -lm %%a -lzm %%b >> "%LOG%"
        "%BIN%\llama-bench.exe" -m "%MODEL%" -v -fa on -p 8 -n 0 -ub 512 -r 1 --no-warmup -lm %%a -lzm %%b %EXTRA% >> "%LOG%" 2>&1
        set "EC=!ERRORLEVEL!"
        if not "!EC!"=="0" set "RC=!EC!"
        echo ### exit=!EC! >> "%LOG%"
    )
)

echo.
echo done, %LOG%
exit /b %RC%
