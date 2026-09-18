@echo off
rem throughput of flash attention over a 32k cache with a 2048-cell budget.
rem no model needed.
rem   n_kv_max=0     dense causal mask, the baseline
rem   sparse_blk     length of a run of selected cells (1 = scattered, 4 = qsa blocks)
rem   sparse_grp     query rows sharing one selection (1 = per token, as the model does it)
rem the point of the run: how much the backend gains as blk and grp grow.
rem the gather only skips a key tile when every row of the tile misses it, so the height of the
rem tile decides how much a per-token selection can recover: coopmat needs 16 rows, the scalar
rem path uses 4 to 8. the second pass runs the same cases with coopmat off to measure that.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

set "LOG=%LOGS%\02-fa-perf-%TS%.log"
echo writing %LOG%

"%BIN%\test-backend-ops.exe" perf -o FLASH_ATTN_EXT -p "kv=32768" > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo exit=%RC% >> "%LOG%"

rem the same cases with the pre-pass unioning the rows of the tile, so a per-token selection
rem (sparse_grp=1) also gets a shared list. it only pays off if the rows of a tile overlap
if not defined FAGROUP set "FAGROUP=1"
if "%FAGROUP%"=="1" (
    echo. >> "%LOG%"
    echo ### GGML_VK_FA_SPARSE_GROUP=1, one list per tile >> "%LOG%"
    set "GGML_VK_FA_SPARSE_GROUP=1"
    "%BIN%\test-backend-ops.exe" perf -o FLASH_ATTN_EXT -p "kv=32768" >> "%LOG%" 2>&1
    set "EC=!ERRORLEVEL!"
    set "GGML_VK_FA_SPARSE_GROUP="
    echo exit=!EC! >> "%LOG%"
    if not "!EC!"=="0" set "RC=!EC!"
)

rem the same cases on the scalar path, where a query tile is 4 to 8 rows instead of 16
if not defined FASCALAR set "FASCALAR=1"
if "%FASCALAR%"=="1" (
    echo. >> "%LOG%"
    echo ### GGML_VK_DISABLE_COOPMAT=1, scalar path >> "%LOG%"
    set "GGML_VK_DISABLE_COOPMAT=1"
    "%BIN%\test-backend-ops.exe" perf -o FLASH_ATTN_EXT -p "kv=32768" >> "%LOG%" 2>&1
    set "EC=!ERRORLEVEL!"
    set "GGML_VK_DISABLE_COOPMAT="
    echo exit=!EC! >> "%LOG%"
    if not "!EC!"=="0" set "RC=!EC!"
)

type "%LOG%"
exit /b %RC%
