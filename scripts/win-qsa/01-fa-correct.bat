@echo off
rem correctness of flash attention on the sparse qsa-shaped masks.
rem no model needed. every case must print OK.
setlocal
call "%~dp0_config.bat"

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

set "LOG=%LOGS%\01-fa-correct-%TS%.log"
echo writing %LOG%

"%BIN%\test-backend-ops.exe" test -o FLASH_ATTN_EXT -p "n_kv_max=[1-9]" > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo exit=%RC% >> "%LOG%"

rem prefill folds the heads by default, so the same cases again with the fold off keep the
rem union and the one-row tile covered. the knob is vulkan only, another backend would repeat
if /i not "%BACKEND%"=="vulkan" goto :show
echo. >> "%LOG%"
echo ### GGML_VK_FA_SPARSE_GQA=0 >> "%LOG%"
set "GGML_VK_FA_SPARSE_GQA=0"
"%BIN%\test-backend-ops.exe" test -o FLASH_ATTN_EXT -p "n_kv_max=[1-9]" >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
set "GGML_VK_FA_SPARSE_GQA="
echo exit=%EC% >> "%LOG%"
if not "%EC%"=="0" set "RC=%EC%"

:show
type "%LOG%"
exit /b %RC%
