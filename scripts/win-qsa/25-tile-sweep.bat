@echo off
rem medium quant tile sweep (GGML_VK_MMQ_TILE). on the AMD driver under windows the large matmul tile spills
rem registers and is off, so the medium tile runs every large quant matmul of the prefill, the q8_0
rem projections and the expert matmuls alike. each arm sets the 11 warptile values of that tile
rem (BLOCK_SIZE,BM,BN,BK,WM,WN,WMITER,TM,TN,TK,WARP), checks the quant matmuls against the cpu (q8_0, q4_K and
rem q5_1, MUL_MAT and MUL_MAT_ID) and measures the qwen4exp prefill shapes at 4096 tokens. the first arm is the
rem build default. the throughput of a tile only counts when its tests pass. no model needed. the tiles are
rem TILES in _config.bat
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if /i not "%BACKEND%"=="vulkan" (
    echo backend is %BACKEND%, 25-tile-sweep is vulkan only
    exit /b 0
)

if not exist "%BIN%\test-backend-ops.exe" (
    echo not built: %BIN%\test-backend-ops.exe
    echo run 00-build.bat first
    exit /b 1
)

set "SUM=%LOGS%\25-tile-%TS%-summary.txt"
rem the q8_0 projections of the graph at -ub 4096 and the expert matmuls (q4_K / q4_0 / q5_1 gate and up, q5_1 down)
set "MMP=type_a=q8_0,type_b=f32,m=(320|640|2560|6144|10240|12288),n=4096,k=(320|2560|6144|10240),bs=\[1,1\]"
set "IDP=n_mats=512,n_used=10,b=0,m=(640|2560),n=4096"
set "TQP=type_a=(q8_0|q4_K|q5_1)"
set "RC=0"
set "N=0"
echo writing %SUM%

call :tile ""
for %%t in (%TILES%) do call :tile %%t

set "GGML_VK_MMQ_TILE="

echo.
type "%SUM%"
echo.
echo done, %SUM%
exit /b %RC%

rem %1 = the 11 warptile values in quotes, "" = the build default
:tile
set /a N+=1
set "T=%~1"
set "LABEL=default"
if not "%T%"=="" set "LABEL=%T%"
set "GGML_VK_MMQ_TILE=%T%"
set "TLOG=%LOGS%\25-tile-%TS%-%N%-test.log"
set "PLOG=%LOGS%\25-tile-%TS%-%N%-perf.log"
echo === tile %N%: %LABEL%
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o MUL_MAT -p "%TQP%" > "%TLOG%" 2>&1
set "E1=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" test -b Vulkan0 -o MUL_MAT_ID -p "%TQP%" >> "%TLOG%" 2>&1
set "E2=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT -p "%MMP%" > "%PLOG%" 2>&1
set "E3=%ERRORLEVEL%"
"%BIN%\test-backend-ops.exe" perf -b Vulkan0 -o MUL_MAT_ID -p "%IDP%" >> "%PLOG%" 2>&1
set "E4=%ERRORLEVEL%"
if not "%E1%%E2%%E3%%E4%"=="0000" set "RC=1"
echo ### tile %N%: %LABEL%, exit test %E1% %E2% perf %E3% %E4% >> "%SUM%"
findstr /C:"medium quant tile" /C:"ignored" /C:"tests passed" "%TLOG%" >> "%SUM%"
findstr /C:"MUL_MAT(" /C:"MUL_MAT_ID(" "%PLOG%" >> "%SUM%"
echo. >> "%SUM%"
goto :eof
