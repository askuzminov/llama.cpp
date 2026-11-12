@echo off
rem the build, the upstream build of 07 and every step with the variants of the open questions, in order: 00-build,
rem 00-build upstream, 04-micro (off), 01-check, 06-decode, 03-profile, 07-kld, 07-kld against the exact f32 base
rem (the 395 only), 02-long. about 3 hours on the 395, more on the 3090 (two CUDA builds, more variants; README).
rem each step runs only what the same sources have not measured yet, so after a change of the code everything runs
rem and after a change of the variants only the new ones. the variants below are set only where _local.bat has none;
rem RUN_BUILD, RUN_UPSTREAM, RUN_MICRO, RUN_CHECK, RUN_DECODE, RUN_PROFILE, RUN_KLD, RUN_EXACT, RUN_LONG=0 there
rem leave a step out. serve.bat is not part of it and has to be stopped
setlocal enabledelayedexpansion
rem the folder of this file, for the subroutine below: there the zeroth argument is the label, not this file
set "HERE=%~dp0"
call "%HERE%_config.bat"

rem the questions after the runs of 12.11 (README, run-all): the guards of the device memory (01), the QSA block lists
rem of the slots (02 par, rows0), the PPL gap against upstream (+1.0% on the 395, 12.11: not the PLE rows, the QSA pool,
rem the attention list or the QSA selection; about 40% our graph, the rest also with the graph of upstream in this
rem build): on the 395 07 with the graph of upstream without QSA selection, the fusions of Vulkan off, the radix top-k
rem of upstream; on the 3090 the same split on CUDA and the whole model on the CPU. 07-exact on the 395 keeps the prompt
rem path and the decode kernels at 4 tokens against the exact f32 math (a regression check; f32 mat-vecs and 8 tokens
rem answered 12.11). on the 3090 the expert cache after the fix of the generation phase against the expert cache of
rem upstream with a CPU share of the misses (upmoe5, upmoe8: 06; they did not run on 12.11) and the radix top-k of
rem upstream (radix-up). settled and left out: on the 395 mm and radix0
rem (01.11, 02.11); on the 3090 nowin (02.11), and 12.11 cap7 (-1.1%), full (the whole vocabulary of the MTP head,
rem -0.3%), m256, upmoe0 (-1.6%), upmoe3, radix-up in 03 (the prompt within 1.5%), m512 (the card ran full and the
rem guard refused 256 bytes: 512 MiB next to the expert cache is too little); again (a second default) stays: the
rem logits of default bit for bit
if not defined QA_QARMS set "QA_QARMS=default;again LLAMA_QA_AGAIN=1"
if not defined QA_PAR_ARMS set "QA_PAR_ARMS=default;rows0 LLAMA_QSA_ROWS=0"
rem the decode tasks with a long prompt (about 24K tokens) and a short reply: the prompt speed of each variant too
if not defined QA_DEC_SET set "QA_DEC_SET=prose ru code-py code-cpp edit json long"
rem the probes of 02-long take the default only, the variants of the probes are a question for 01-check
if not defined LONG_QARMS set "LONG_QARMS=default"
rem the indexer kernels have not changed since the micro of 31.10 evening: 04-micro only on request (RUN_MICRO=1)
if not defined RUN_MICRO set "RUN_MICRO=0"
if /i "%BACKEND%"=="cuda" goto :arms_cuda
if not defined QA_PARMS set "QA_PARMS=default"
if not defined QA_DARMS set "QA_DARMS=default"
if not defined QA_KARMS set "QA_KARMS=fork;copy LLAMA_UPSTREAM=qwen4exp;dense "QA_ARM_ARGS=--override-kv qwen4exp.attention.indexer.top_k=int:262144";copydense LLAMA_UPSTREAM=qwen4exp "QA_ARM_ARGS=--override-kv qwen4exp.attention.indexer.top_k=int:262144";nofuse GGML_VK_DISABLE_FUSION=1;copynofuse LLAMA_UPSTREAM=qwen4exp GGML_VK_DISABLE_FUSION=1;radix0 GGML_VK_TOPK_RADIX_STABLE=0;upstream @upstream;updense @upstream "QA_ARM_ARGS=--override-kv qwen4exp.attention.indexer.top_k=int:262144""
if not defined EXACT_KARMS set "EXACT_KARMS=prompt;decode4 QA_ARM_UB=4"
goto :arms_done
:arms_cuda
if not defined QA_PARMS set "QA_PARMS=default"
rem upmoe: 6500 MiB, about the ~46 slots a layer of our cache at full depth (RESEARCH.md); --moe-cache 0 turns ours
rem off. 12.11 the share of the misses on the CPU: 0 -1.6%, 0.3 +7.4%, 0.5 +20.3% TG, the 22K prompt 33% slower
rem (all against our cache with the buffers of the prompt phase kept, before the fix)
if not defined QA_DARMS set "QA_DARMS=default;upmoe5 "QA_ARM_ARGS=--moe-cache 0 --moe-cache-mib 6500 --moe-cache-cpu-share 0.5";upmoe8 "QA_ARM_ARGS=--moe-cache 0 --moe-cache-mib 6500 --moe-cache-cpu-share 0.8""
rem cpu, upcpu: the whole model on the CPU (128 GB here, the 395 has 32 GB beside its GPU): our code against upstream
rem without either GPU backend
if not defined QA_KARMS set "QA_KARMS=fork;upstream @upstream;copy LLAMA_UPSTREAM=qwen4exp;lzmon "QA_ARM_ARGS=-lzm on";dense "QA_ARM_ARGS=--override-kv qwen4exp.attention.indexer.top_k=int:262144";updense @upstream "QA_ARM_ARGS=--override-kv qwen4exp.attention.indexer.top_k=int:262144";cpu "QA_ARM_ARGS=-ngl 0 -dev none";upcpu @upstream "QA_ARM_ARGS=-ngl 0 -dev none";radix-up GGML_CUDA_TOPK_RADIX_STABLE=0"
rem the CUDA matmuls have no f32 mode: 07 against the exact base runs on the 395 only
if not defined RUN_EXACT set "RUN_EXACT=0"
:arms_done

for %%v in (BUILD MICRO CHECK DECODE PROFILE KLD EXACT LONG) do if not defined RUN_%%v set "RUN_%%v=1"
rem the upstream build is the base of 07 against upstream
if not defined RUN_UPSTREAM set "RUN_UPSTREAM=%RUN_KLD%"
set "STEPS="
if "%RUN_BUILD%"=="1"    set "STEPS=!STEPS! 00-build"
if "%RUN_UPSTREAM%"=="1" set "STEPS=!STEPS! 00-upstream"
if "%RUN_MICRO%"=="1"    set "STEPS=!STEPS! 04-micro"
if "%RUN_CHECK%"=="1"    set "STEPS=!STEPS! 01-check"
if "%RUN_DECODE%"=="1"   set "STEPS=!STEPS! 06-decode"
if "%RUN_PROFILE%"=="1"  set "STEPS=!STEPS! 03-profile"
if "%RUN_KLD%"=="1"      set "STEPS=!STEPS! 07-kld"
if "%RUN_EXACT%"=="1"    set "STEPS=!STEPS! 07-exact"
if "%RUN_LONG%"=="1"     set "STEPS=!STEPS! 02-long"
if "%STEPS%"=="" (
    echo every RUN_ step is 0, nothing to do
    exit /b 1
)

rem the probes of 01 and 02, the decode tasks and 07 read a long text
if not defined QA_CORPUS if not exist "%PPLFILE%" call "%HERE%get-wikitext.bat"

echo steps:%STEPS%
echo variants: micro [%QA_MARMS%] check [%QA_QARMS%] decode [%QA_DARMS%] profile [%QA_PARMS%] kld [%QA_KARMS%]
echo   exact [%EXACT_KARMS%] long [%LONG_QARMS%] par [%QA_PAR_ARMS%], upstream %UPSTREAMREF%
rem 00-build waits for a key after a failure, run-all goes on by itself
set "QSA_UNATTENDED=1"
rem not RC: cmake takes RC from the environment as the resource compiler of a fresh configure
set "STEPRC=0"
set "RESULTS="
set "CHECK_QARMS=%QA_QARMS%"
set "KLD_KARMS=%QA_KARMS%"
for %%s in (%STEPS%) do (
    echo === %%s
    call :step %%s
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

rem one step with its variants: 02-long reads the probes of LONG_QARMS, 07-exact compares with the exact f32 base. a
rem failed 00-upstream leaves 07 against upstream to its check of the build (UPSTREAMREF), the other steps go on
:step
set "QA_QARMS=%CHECK_QARMS%"
set "QA_KARMS=%KLD_KARMS%"
set "QA_KLD_BASE=upstream"
if "%~1"=="00-upstream" goto :step_upstream
if "%~1"=="07-exact" goto :step_exact
if "%~1"=="02-long" set "QA_QARMS=%LONG_QARMS%"
call "%HERE%%~1.bat"
exit /b %ERRORLEVEL%
:step_upstream
call "%HERE%00-build.bat" upstream
exit /b %ERRORLEVEL%
:step_exact
set "QA_KLD_BASE=exact"
set "QA_KARMS=%EXACT_KARMS%"
call "%HERE%07-kld.bat"
exit /b %ERRORLEVEL%
