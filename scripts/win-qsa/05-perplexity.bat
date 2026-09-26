@echo off
rem perplexity of the current build, the reference number for the model. it only means
rem something when the context is well above the qsa budget of about 2051 cells: below it
rem the indexer is bypassed and the run says nothing about the selection.
rem each value of FAVARIANTS is one arm, see 04-fa-sparse.bat: 1 the build default (the head
rem fold on prefill), 0 dense, g the gather over the union of the rows of the tile, r the
rem gather over a one-row tile, g and r with the head fold off.
rem every arm walks the same cells, so all of them have to land on the same number. 06 tells
rem how far the kernels sit from each other, this one tells how far each sits from the
rem text, which is the number that says whether quality is kept.
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not defined QCTX    set "QCTX=32768"
if not defined QCHUNKS set "QCHUNKS=4"
if not defined FAVARIANTS set "FAVARIANTS=1 0 g"

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

if not exist "%PPLFILE%" (
    echo missing %PPLFILE%, run get-wikitext.bat first
    exit /b 1
)

set "SUM=%LOGS%\05-ppl-%TS%-summary.txt"
set "RC=0"

for %%v in (%FAVARIANTS%) do call :run %%v

set "GGML_VK_FA_SPARSE_DISABLE="
set "GGML_VK_FA_SPARSE_GROUP="
set "GGML_VK_FA_SPARSE_GQA="

echo.
type "%SUM%"
exit /b %RC%

rem %1 = плечо, как в 04: 0 плотное ядро, 1 умолчание, g объединение, r тайл в одну строку
:run
set "LOG=%LOGS%\05-ppl-%TS%-s%~1.log"
set "GGML_VK_FA_SPARSE_DISABLE="
set "GGML_VK_FA_SPARSE_GROUP="
set "GGML_VK_FA_SPARSE_GQA="
if "%~1"=="0" set "GGML_VK_FA_SPARSE_DISABLE=1"
if "%~1"=="g" set "GGML_VK_FA_SPARSE_GROUP=1"
if "%~1"=="r" set "GGML_VK_FA_SPARSE_GROUP=0"
if "%~1"=="g" set "GGML_VK_FA_SPARSE_GQA=0"
if "%~1"=="r" set "GGML_VK_FA_SPARSE_GQA=0"
echo === perplexity, sparse=%~1 c=%QCTX% chunks=%QCHUNKS% -^> %LOG%
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c %QCTX% --chunks %QCHUNKS% -fa on %LOADMODE% %EXTRA% > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo. >> "%SUM%"
echo ### sparse=%~1 c=%QCTX% chunks=%QCHUNKS% exit=%EC% >> "%SUM%"
findstr /c:"Final estimate" "%LOG%" >> "%SUM%"
goto :eof
