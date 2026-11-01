@echo off
rem the build and every step with the variants of the open questions, in order: 00-build, 04-micro, 01-check,
rem 03-profile, 02-long (about two and a half hours on the 395). each step runs only what the same sources have not
rem measured yet, so after a change of the code everything runs and after a change of the variants only the new
rem ones. the variants below are set only where _local.bat has none; RUN_BUILD, RUN_MICRO, RUN_CHECK, RUN_PROFILE,
rem RUN_LONG=0 there leave a step out. serve.bat is not part of it and has to be stopped
setlocal enabledelayedexpansion
call "%~dp0_config.bat"

rem the questions of 01.11 (README). the 395: the lightning indexer as the default of the Vulkan graph (mm: the head
rem matmul in the graph, the default before) and the radix-select of the block top-k in index order (radix0: the
rem atomic order of upstream). check: again (a second default, an unread variable) and mm give the logits of default
rem bit for bit, radix0 not past the first ubatch of 4096 tokens; profile: the time and compute buffer of each. the
rem 3090: again, its radix-select (CUDA top-k.cu) still compacts with atomics; the MTP window KV on by default since
rem 01.11 also with one slot (nowin: the default before)
if /i not "%BACKEND%"=="vulkan" goto :arms_cuda
if not defined QA_QARMS set "QA_QARMS=default;again LLAMA_QA_AGAIN=1;mm LLAMA_QSA_LID=0;radix0 GGML_VK_TOPK_RADIX_STABLE=0"
if not defined QA_PARMS set "QA_PARMS=default;mm LLAMA_QSA_LID=0;radix0 GGML_VK_TOPK_RADIX_STABLE=0"
goto :arms_done
:arms_cuda
if not defined QA_QARMS set "QA_QARMS=default;again LLAMA_QA_AGAIN=1"
if not defined QA_PARMS set "QA_PARMS=default;nowin LLAMA_MTP_WINDOW_KV=-1"
:arms_done
rem the indexer kernels have not changed since the micro of 31.10 evening: 04-micro only on request (RUN_MICRO=1)
if not defined RUN_MICRO set "RUN_MICRO=0"
rem the probes of 02-long take the default only, the variants of the probes are a question for 01-check
if not defined LONG_QARMS set "LONG_QARMS=default"

for %%v in (BUILD MICRO CHECK PROFILE LONG) do if not defined RUN_%%v set "RUN_%%v=1"
set "STEPS="
if "%RUN_BUILD%"=="1"   set "STEPS=!STEPS! 00-build"
if "%RUN_MICRO%"=="1"   set "STEPS=!STEPS! 04-micro"
if "%RUN_CHECK%"=="1"   set "STEPS=!STEPS! 01-check"
if "%RUN_PROFILE%"=="1" set "STEPS=!STEPS! 03-profile"
if "%RUN_LONG%"=="1"    set "STEPS=!STEPS! 02-long"
if "%STEPS%"=="" (
    echo every RUN_ step is 0, nothing to do
    exit /b 1
)

rem the probes of 01 and 02 read a long text
if not defined QA_CORPUS if not exist "%PPLFILE%" call "%~dp0get-wikitext.bat"

echo variants: micro [%QA_MARMS%] check [%QA_QARMS%] profile [%QA_PARMS%] long [%LONG_QARMS%]
rem 00-build waits for a key after a failure, run-all goes on by itself
set "QSA_UNATTENDED=1"
rem not RC: cmake takes RC from the environment as the resource compiler of a fresh configure
set "STEPRC=0"
set "RESULTS="
set "CHECK_QARMS=%QA_QARMS%"
for %%s in (%STEPS%) do (
    echo === %%s
    if "%%s"=="02-long" (set "QA_QARMS=%LONG_QARMS%") else (set "QA_QARMS=%CHECK_QARMS%")
    call "%~dp0%%s.bat"
    set "EC=!ERRORLEVEL!"
    if not "!EC!"=="0" set "STEPRC=!EC!"
    set "RESULTS=!RESULTS! %%s=!EC!"
    echo.
    rem the steps after a failed build would measure the old binaries
    if "%%s"=="00-build" if not "!EC!"=="0" goto :done
)

:done
echo === exit codes:%RESULTS%
echo the summary of each step is in its folder in %LOGS%
exit /b %STEPRC%
