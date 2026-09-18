@echo off
rem how big the compute buffer gets as the context grows. llama-server at -c 262144 -ub 2048
rem asked the vulkan allocator for 19.13 GiB and died, so this walks both axes and writes the
rem reserved size of each arm. the reserve is worst case: n_kv is the whole cache and n_tokens
rem is min(n_ctx, n_ubatch), so the size grows with the product of the two.
rem llama-fit-params reads the metadata and simulates the allocation, so no weights are read
rem and an arm takes about a second.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem one arm per line, "ctx ubatch"
if not defined CTXVARIANTS set "CTXVARIANTS="262144 2048" "262144 1024" "262144 512" "131072 2048" "131072 512" "65536 512" "32768 512""

if not exist "%BIN%\llama-fit-params.exe" (
    echo not built: %BIN%\llama-fit-params.exe
    echo run 00-build.bat first
    exit /b 1
)

if not exist "%MODEL%" (
    echo model not found: %MODEL%
    echo set MODEL in _local.bat
    exit /b 1
)

set "SUM=%LOGS%\16-ctx-%TS%-summary.txt"
echo ### 16-ctx %TS% > "%SUM%"
echo ### MODEL=%MODEL% >> "%SUM%"
echo ### EXTRA=%EXTRA% >> "%SUM%"
echo ### columns: device model context compute, MiB >> "%SUM%"
echo writing %SUM%

set "RC=0"

for %%v in (%CTXVARIANTS%) do (
    for /f "tokens=1,2" %%a in (%%v) do (
        set "LOG=%LOGS%\16-ctx-%TS%-c%%a-ub%%b.log"
        echo === -c %%a -ub %%b -^> !LOG!
        "%BIN%\llama-fit-params.exe" -m "%MODEL%" -c %%a -ub %%b -fa on -np 1 -fitp on %EXTRA% > "!LOG!" 2>&1
        set "EC=!ERRORLEVEL!"

        echo. >> "%SUM%"
        echo ### -c %%a -ub %%b exit=!EC! >> "%SUM%"
        findstr /r /c:"^[A-Za-z]" /c:"failed to allocate" /c:"exceeds device buffer size" "!LOG!" >> "%SUM%"

        if not "!EC!"=="0" set "RC=!EC!"
    )
)

echo.
echo === summary
type "%SUM%"
exit /b %RC%
