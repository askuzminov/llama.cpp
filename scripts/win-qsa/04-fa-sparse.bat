@echo off
rem the same prefill sweep over every shape of the sparse flash attention path, one log per
rem arm plus a summary. every arm walks the same cells, so this is a pure speed question:
rem does the gather beat the dense kernel on a real per-token mask, and which tile shape
rem does it best. on prefill the gather has two shapes. the union keeps the query tile as it
rem is and walks the cells any row of the tile keeps, so the coopmat matmul stays but the
rem list grows with the tile. the one-row tile walks an exact list of n_kv_max cells and
rem gives up the coopmat matmul. arm 1 is the build default, which picks between them.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem плечи: 0 плотное ядро, 1 умолчание сборки, g объединение строк тайла, r только тайл
rem в одну строку. g минус r это и есть ответ, какая форма тайла быстрее, а 1 показывает,
rem ту ли из них выбирает сборка сама
if not defined FAVARIANTS set "FAVARIANTS=1 0 g r"
rem порог, с которого префил переходит на тайл в одну строку: KV >= ratio * n_kv_max.
rem меряется на плече r, иначе сборка может уйти в объединение и порог ничего не решит.
rem 8 это умолчание сборки и его уже меряет само плечо r, тут только остальные точки
if not defined FARATIOS set "FARATIOS=4 16"

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

if not defined REPS set "REPS=1"

set "SUM=%LOGS%\04-fa-sparse-%TS%-summary.txt"
set "RC=0"

for %%v in (%FAVARIANTS%) do call :run %%v
rem the ratio is a vulkan knob, another backend would just run the same arm again
if /i "%BACKEND%"=="vulkan" for %%r in (%FARATIOS%) do call :run r %%r

set "GGML_VK_FA_SPARSE_DISABLE="
set "GGML_VK_FA_SPARSE_ROW_RATIO="
set "GGML_VK_FA_SPARSE_GROUP="

echo.
echo summary in %SUM%
type "%SUM%"
exit /b %RC%

rem %1 = плечо: 0 плотное ядро, 1 умолчание, g объединение, r тайл в одну строку.
rem %2 = порог тайла в одну строку, только для плеча r
:run
set "TAG=s%~1"
if not "%~2"=="" set "TAG=s%~1-r%~2"
set "LOG=%LOGS%\04-fa-sparse-%TS%-%TAG%.log"
set "GGML_VK_FA_SPARSE_DISABLE="
set "GGML_VK_FA_SPARSE_GROUP="
if "%~1"=="0" set "GGML_VK_FA_SPARSE_DISABLE=1"
if "%~1"=="g" set "GGML_VK_FA_SPARSE_GROUP=1"
if "%~1"=="r" set "GGML_VK_FA_SPARSE_GROUP=0"
set "GGML_VK_FA_SPARSE_ROW_RATIO=%~2"
echo === sparse=%~1 ratio=%~2 -^> %LOG%
"%BIN%\llama-bench.exe" -m "%MODEL%" -fa on -p %NPROMPT% -n 0 -b 4096 -ub %UBATCH% -d %DEPTHS% %LOADMODE% -r %REPS% %EXTRA% --progress -o md > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo. >> "%SUM%"
echo ### sparse=%~1 ratio=%~2 exit=%EC% >> "%SUM%"
type "%LOG%" >> "%SUM%"
goto :eof
