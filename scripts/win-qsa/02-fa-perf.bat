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

rem the run above takes whichever tile shape the build picks: on this shape (12 q heads per kv
rem head) that is the head fold, one query row over the heads of a group with an exact list.
rem these two turn the fold off and force each of the older shapes, so the log holds all
rem three. the union gives a per-token selection (sparse_grp=1) a shared list and keeps the
rem coopmat matmul, and only pays off if the rows of a tile overlap. the one-row tile gets an
rem exact list and gives up the matmul
if not defined FAGROUP set "FAGROUP=1"
rem both arms and the scalar one below force a vulkan pipeline, another backend would just
rem repeat the run above
if /i not "%BACKEND%"=="vulkan" set "FAGROUP=0"
if "%FAGROUP%"=="1" (
    call :arm 1 "one list per tile"
    call :arm 0 "one row per tile"
)

rem the same cases on the scalar path, where a query tile is 4 to 8 rows instead of 16
if not defined FASCALAR set "FASCALAR=1"
if /i not "%BACKEND%"=="vulkan" set "FASCALAR=0"
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

rem %1 = value of GGML_VK_FA_SPARSE_GROUP, %2 = what it means in the log
:arm
echo. >> "%LOG%"
echo ### GGML_VK_FA_SPARSE_GQA=0 GGML_VK_FA_SPARSE_GROUP=%~1, %~2 >> "%LOG%"
set "GGML_VK_FA_SPARSE_GQA=0"
set "GGML_VK_FA_SPARSE_GROUP=%~1"
"%BIN%\test-backend-ops.exe" perf -o FLASH_ATTN_EXT -p "kv=32768" >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_FA_SPARSE_GROUP="
set "GGML_VK_FA_SPARSE_GQA="
echo exit=%EC% >> "%LOG%"
if not "%EC%"=="0" set "RC=%EC%"
goto :eof
