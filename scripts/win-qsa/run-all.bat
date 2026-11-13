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

rem the questions of 13.11 night (README, run-all). the 395: the precision of the draft checks, dense q8_0 of 2-8
rem columns. vfyp2 (the f16 coopmat tile of GGML_VK_MM_F16REG_VERIFY for 1024 rows and more, two int8 planes below):
rem KLD 0.0115 against 0.0118 without drafts and 0.0311 of default; 13.11 22:00 (e194f0bfb, read in place, r16x16/4x1)
rem 06 TG -0.6% (the f16 copy -0.5%), but in 03 a verify of the same size +1.6-2.7 ms: the tile reads the big
rem matrices at 190-202 GB/s in the model against 207-235 of the int8 mat-vec. now the f32 mat-vec that exists
rem (f32d, GGML_VK_DISABLE_MMVQ=1: dense q8_0 against f32 activations, the experts stay on the fork mat-vec): 03 and 06
rem default, vfyp2, f32d and again (default once more at the end: the order effect, 13.11 the draft step of the first
rem arm 1.3-2.7% slower), all measured anew in this run (FORCE); 07-exact f32d; 01 only the spec probe again (default,
rem vfyp2, f32d; the PIT mean of the mid-point transform). 05, 07 against upstream and 02 are off on the 395. the 3090: 01 (the MUL_MAT_VERIFY cases on CUDA, in its CHECKOPS since 13.11;
rem 13.11 21:36 all ops passed without them); 06, 07, 03 and 02 are off (03 answered 13.11: the reply 38.96-46.31 ms of
rem GPU a token with the reports read per backend context). both: the spec probe of 01 again, the PIT mean now of the
rem mid-point transform (13.11: the randomized one took a fixed random stream whose first 1500 values average 0.516, so
rem the mean without drafts was 0.511-0.518 in every run). settled and left out: on the 395 mm and radix0 (01.11, 02.11), the 07 split of 12.11 (copy,
rem dense, nofuse, radix0); on the 3090 nowin (02.11), and 12.11 cap7 (-1.1%), full (the whole vocabulary of the MTP
rem head, -0.3%), m256, upmoe0 (-1.6%), upmoe3, radix-up in 03 (the prompt within 1.5%), m512 (the card ran full and
rem the guard refused 256 bytes: 512 MiB next to the expert cache is too little); 13.11 upmoe5/upmoe8 (the cache of
rem upstream -20%/-10% TG in two runs) and the 07 split (copy = upstream bit for bit, the CPU: fork 1.9434 against
rem upstream 1.9442); again (a second default) stays: the logits of default bit for bit
if not defined QA_PAR_ARMS set "QA_PAR_ARMS=default;rows0 LLAMA_QSA_ROWS=0"
rem the decode tasks with a long prompt (about 24K tokens) and a short reply: the prompt speed of each variant too
if not defined QA_DEC_SET set "QA_DEC_SET=prose ru code-py code-cpp edit json long"
rem the probes of 02-long take the default only, the variants of the probes are a question for 01-check
if not defined LONG_QARMS set "LONG_QARMS=default"
rem the indexer kernels have not changed since the micro of 31.10 evening: 04-micro only on request (RUN_MICRO=1)
if not defined RUN_MICRO set "RUN_MICRO=0"
if /i "%BACKEND%"=="cuda" goto :arms_cuda
if not defined QA_QARMS set "QA_QARMS=default"
if not defined QA_SPEC_ARMS set "QA_SPEC_ARMS=default;vfyp2 GGML_VK_MM_F16REG_VERIFY=1 GGML_VK_MMVQ_PLANES=2 GGML_VK_MMVQ_PLANES_TYPES=q8_0;f32d GGML_VK_DISABLE_MMVQ=1"
if not defined QA_PARMS set "QA_PARMS=default;vfyp2 GGML_VK_MM_F16REG_VERIFY=1 GGML_VK_MMVQ_PLANES=2 GGML_VK_MMVQ_PLANES_TYPES=q8_0;f32d GGML_VK_DISABLE_MMVQ=1;again LLAMA_QA_AGAIN=1"
rem 03: depth 0 only (the dense matmuls do not depend on it), a reply of 256 tokens for verifies of every size, the
rem tables of the ops at a verify of 4 tokens
if not defined QA_PROF_DEPTHS set "QA_PROF_DEPTHS=0"
if not defined QA_PROF_TG set "QA_PROF_TG=256"
if not defined QA_PROF_VERIFY set "QA_PROF_VERIFY=4"
if not defined QA_DARMS set "QA_DARMS=default;vfyp2 GGML_VK_MM_F16REG_VERIFY=1 GGML_VK_MMVQ_PLANES=2 GGML_VK_MMVQ_PLANES_TYPES=q8_0;f32d GGML_VK_DISABLE_MMVQ=1;again LLAMA_QA_AGAIN=1"
rem default and vfyp2 of 03 and 06 were measured on these sources at 22:00 13.11: all variants again in one session
if not defined FORCE set "FORCE=^(decode|profile)"
if not defined QA_KARMS set "QA_KARMS=fork;upstream @upstream"
if not defined EXACT_KARMS set "EXACT_KARMS=f32d QA_ARM_UB=4 GGML_VK_DISABLE_MMVQ=1"
if not defined RUN_KLD set "RUN_KLD=0"
if not defined RUN_LONG set "RUN_LONG=0"
goto :arms_done
:arms_cuda
if not defined QA_QARMS set "QA_QARMS=default;again LLAMA_QA_AGAIN=1"
if not defined QA_PARMS set "QA_PARMS=default"
if not defined QA_DARMS set "QA_DARMS=default"
if not defined QA_KARMS set "QA_KARMS=fork;upstream @upstream"
if not defined RUN_DECODE set "RUN_DECODE=0"
if not defined RUN_PROFILE set "RUN_PROFILE=0"
if not defined RUN_KLD set "RUN_KLD=0"
if not defined RUN_LONG set "RUN_LONG=0"
rem the CUDA matmuls have no f32 mode: 07 against the exact base runs on the 395 only
if not defined RUN_EXACT set "RUN_EXACT=0"
:arms_done
rem 05-mm (the dense q8_0 matmuls by tile) only where a section above or _local.bat asks (RUN_MM=1)
if not defined RUN_MM set "RUN_MM=0"

for %%v in (BUILD MICRO CHECK DECODE PROFILE KLD EXACT LONG) do if not defined RUN_%%v set "RUN_%%v=1"
rem the upstream build is the base of 07 against upstream
if not defined RUN_UPSTREAM set "RUN_UPSTREAM=%RUN_KLD%"
set "STEPS="
if "%RUN_BUILD%"=="1"    set "STEPS=!STEPS! 00-build"
if "%RUN_UPSTREAM%"=="1" set "STEPS=!STEPS! 00-upstream"
if "%RUN_MICRO%"=="1"    set "STEPS=!STEPS! 04-micro"
if "%RUN_MM%"=="1"       set "STEPS=!STEPS! 05-mm"
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
