@echo off
rem how far the sparse flash attention path moves the output distribution away from the dense
rem kernel. both walk the same cells, so the answer should be near zero and this is a check of
rem the gather, not of the selection. the base runs with the gather off, and every value of
rem FAVARIANTS is then measured against it - 0 included, which repeats the base and gives the
rem noise floor of two runs of the same code. a KLD of the gather that is not far above that
rem floor is rounding, one that is above it is a defect.
rem run 05 first if you only want a cheap number: this one writes a logits file of about
rem 300 KB per token.
rem   KLCTX x KLCHUNKS tokens x n_vocab x 2 bytes
rem   32768 x 1 x 151936 x 2 = about 10 GB
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

if not defined FAVARIANTS set "FAVARIANTS=1 0"

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

rem the context has to stay well above the qsa budget, see QCTX in _config.bat
if not defined KLCTX    set "KLCTX=32768"
if not defined KLCHUNKS set "KLCHUNKS=1"
set "BASEFILE=%LOGS%\kl-base-%TS%.dat"

if not exist "%PPLFILE%" (
    echo missing %PPLFILE%, run get-wikitext.bat first
    exit /b 1
)

echo the base logits file will be written to %BASEFILE%
echo expect roughly 10 GB of disk for KLCTX=%KLCTX% KLCHUNKS=%KLCHUNKS%
rem run-all.bat has already decided, do not stop there
if not defined QSA_UNATTENDED pause

set "GGML_VK_FA_SPARSE_DISABLE=1"
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c %KLCTX% --chunks %KLCHUNKS% -fa on %LOADMODE% %EXTRA% --kl-divergence-base "%BASEFILE%" > "%LOGS%\06-kl-%TS%-base.log" 2>&1
if errorlevel 1 (
    echo base run failed, see %LOGS%\06-kl-%TS%-base.log
    exit /b 1
)

set "SUM=%LOGS%\06-kl-%TS%-summary.txt"
set "RC=0"

for %%v in (%FAVARIANTS%) do call :run %%v

set "GGML_VK_FA_SPARSE_DISABLE="

del "%BASEFILE%"

echo.
type "%SUM%"
exit /b %RC%

rem %1 = 1 gather on, 0 gather off. 0 repeats the base and measures the noise floor
:run
set "LOG=%LOGS%\06-kl-%TS%-s%~1.log"
set "GGML_VK_FA_SPARSE_DISABLE="
if "%~1"=="0" set "GGML_VK_FA_SPARSE_DISABLE=1"
echo === sparse=%~1 -^> %LOG%
"%BIN%\llama-perplexity.exe" -m "%MODEL%" -f "%PPLFILE%" -c %KLCTX% --chunks %KLCHUNKS% -fa on %LOADMODE% %EXTRA% --kl-divergence --kl-divergence-base "%BASEFILE%" > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" set "RC=%EC%"
echo. >> "%SUM%"
echo ### sparse=%~1 exit=%EC% >> "%SUM%"
findstr /c:"KL divergence" /c:"Same top" /c:"Mean PPL ratio" /c:"Mean PPL(Q)/PPL(base)" /c:"Mean    KLD" /c:"RMS" "%LOG%" >> "%SUM%"
goto :eof
