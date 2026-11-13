@echo off
rem the settings of the scripts, the same on every machine. _local.bat next to this file (ignored by git, start from
rem _local.example.bat) sets MODEL, SPECDRAFT and what differs on that machine; it runs after the first block here,
rem and the defaults after it keep what it set. README.md describes every value.
rem git bash and msys put their own find, sort and friends ahead of the windows ones on PATH
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%PATH%"

set "MODEL=C:\models\qwen3-next.gguf"
set "SPECDRAFT="
set "PPLFILE=%~dp0wikitext-2-raw\wiki.test.raw"
set "BUILD=%~dp0..\..\build-win"
set "CMAKE_BACKEND=auto"
set "GENERATOR=auto"
rem the weights read at load, the PLE rows from the drive on demand (row cache)
set "LOADMODE=-lm dio -lzm dio"
rem seconds after a model run when the gpu memory cannot be read (see settle in _qa.py)
set "SETTLE=30"
if exist "%~dp0_local.bat" call "%~dp0_local.bat"
for %%i in ("%BUILD%") do set "BUILD=%%~fi"

rem backend for cmake. auto picks from what the machine has: nvcc or CUDA_PATH for cuda, VULKAN_SDK or glslc for vulkan
if /i not "%CMAKE_BACKEND%"=="auto" goto :backend_set
set "CMAKE_BACKEND="
where nvcc >nul 2>&1
if not errorlevel 1 set "CMAKE_BACKEND=-DGGML_CUDA=ON"
if defined CMAKE_BACKEND goto :backend_set
if defined CUDA_PATH set "CMAKE_BACKEND=-DGGML_CUDA=ON"
if defined CMAKE_BACKEND goto :backend_set
if defined VULKAN_SDK set "CMAKE_BACKEND=-DGGML_VULKAN=ON"
if defined CMAKE_BACKEND goto :backend_set
where glslc >nul 2>&1
if not errorlevel 1 set "CMAKE_BACKEND=-DGGML_VULKAN=ON"
:backend_set

set "BACKEND=cpu"
rem as in 00-build.bat: a replace in an undefined CMAKE_BACKEND breaks the line before its if runs
if not defined CMAKE_BACKEND goto :backend_named
if not "%CMAKE_BACKEND:GGML_VULKAN=%"=="%CMAKE_BACKEND%" set "BACKEND=vulkan"
if not "%CMAKE_BACKEND:GGML_CUDA=%"=="%CMAKE_BACKEND%" set "BACKEND=cuda"
:backend_named

if /i "%EXTRA%"=="auto" set "EXTRA="
if /i "%BACKEND%"=="cuda" goto :cuda_defaults

rem the 395 (vulkan): the whole model on the gpu. DECARGS: the server as serve.bat runs it (models.ini)
if not defined EXTRA     set "EXTRA=-ngl 99"
if not defined DECARGS   set "DECARGS=-fit off -b 16384 -ub 4096"
rem seconds of rest after a build or a long model run (COOL_RUN seconds, 300) before a timed run: the 395 runs warm
rem (95 C in an hour of chats); the short arms of a profile at depth 0 go one after another
if not defined COOL      set "COOL=120"
if not defined CHECKOPS  set "CHECKOPS=MUL_MAT;MUL_MAT_ID;MUL_MAT_ID_FUSION;MUL_MAT_FUSION;MUL_MAT_RELU;MUL_MAT_HEADSUM;TOP_K;TOPK_QSA;LIGHTNING_INDEXER;LIGHTNING_INDEXER_UNIT;GET_ROWS;GATED_DELTA_NET;GATED_DELTA_NET_CACHE_FUSION;DSV4_HC_PRE;DSV4_HC_POST;HC_POST_GATE;REPEAT;FLASH_ATTN_EXT n_kv_max=[1-9];RMS_NORM;RMS_NORM_SCALE;RMS_NORM_MUL_SIGMOID_MUL;SSM_CONV;CONCAT_SSM_CONV;MUL_MAT_ID_REPEAT;GATED_DELTA_NET_LAZY;FLASH_ATTN_LIST;DSV4_HC_PRE_F16;DSV4_HC_POST_NORM_F16;MOE_GLU_F16_DOWN;MUL_MAT_HC_PRE;HC_MIX;HC_POST_MIX;MUL_MAT_ID_SCALE;MUL_MAT_F16_RANGE;MUL_MAT_ID_F16_RANGE;MUL_MAT_VERIFY"
rem int8: the int8 coopmat path of the experts, off by default (f16reg), checked as well; glu0: the fused gate/up of
rem f16reg (and its f16 result into the down projection) on the short tests too, it runs from 2048 tokens
if not defined CHECKARMS set "CHECKARMS=default;int8 GGML_VK_MMID_F16REG=0 GGML_VK_MMID_GLU_MIN_TOKENS=0 @o=MUL_MAT_ID,MUL_MAT_ID_REPEAT,MUL_MAT_ID_FUSION,MUL_MAT_FUSION,MOE_GLU_F16_DOWN;glu0 GGML_VK_MMID_GLU_MIN_TOKENS=0 @o=MUL_MAT_FUSION,MOE_GLU_F16_DOWN;vfy GGML_VK_MM_F16REG_VERIFY=1 @o=MUL_MAT,MUL_MAT_VERIFY,MUL_MAT_F16_RANGE"
goto :defaults_done

:cuda_defaults
rem the 3090 (cuda): the probe runs keep all 48 MoE layers on the cpu as the server (DECARGS) does: a prompt batch
rem uploads the experts (op offload), a small batch runs them on the cpu. the server also runs the experts of its
rem cache on the card (--moe-cache, small batches only); the probes go without it, so their decode rows run every
rem expert on the cpu. layers on the card do not fit: about 1.5 GiB a layer, -ncmoe 30 put 28 GiB on the 24 of the
rem card, the driver moved the rest to system memory (sysmem fallback) and the probes ran 25 times slower (32768
rem tokens: 26 s with 46 on 31.10, 600-700 s with 30 since), a long run did not end. since 12.11 the backend refuses
rem such an allocation (ggml-cuda/fork/device-room-fork.inc) and the load fails instead
if not defined EXTRA     set "EXTRA=-ngl 99 -ncmoe 48"
if not defined DECARGS   set "DECARGS=-fit off -ncmoe 48 --moe-cache auto --phase-mem --no-repack -b 16384 -ub 8192"
if not defined CHECKOPS  set "CHECKOPS=MUL_MAT;MUL_MAT_ID;MUL_MAT_ID_FUSION;MUL_MAT_FUSION;MUL_MAT_RELU;MUL_MAT_HEADSUM;TOP_K;TOPK_QSA;LIGHTNING_INDEXER;LIGHTNING_INDEXER_UNIT;GET_ROWS;GATED_DELTA_NET;GATED_DELTA_NET_CACHE_FUSION;DSV4_HC_PRE;DSV4_HC_POST;HC_POST_GATE;REPEAT;FLASH_ATTN_EXT n_kv_max=[1-9];RMS_NORM;RMS_NORM_SCALE;RMS_NORM_MUL_SIGMOID_MUL;SSM_CONV;CONCAT_SSM_CONV;MUL_MAT_ID_REPEAT;GATED_DELTA_NET_LAZY;MUL_MAT_F16_RANGE;MUL_MAT_ID_F16_RANGE"
if not defined GGML_OP_OFFLOAD_MIN_BATCH set "GGML_OP_OFFLOAD_MIN_BATCH=128"
if not defined LLAMA_PREFILL_BUDGET set "LLAMA_PREFILL_BUDGET=0"
rem --spec-auto caps drafts at 7 since 09.11 (measured on the 395); the 3090 had 3 ahead on 12.10 (CPU experts:
rem the verify grows with every token) and is not measured since, so its runs keep 3
if not defined QA_SPEC   set "QA_SPEC=--spec-auto --spec-draft-n-max 3"

:defaults_done
rem test-elastic-buffer-fork fails when it tested no device (a skip must not count as a pass)
if not defined TEST_ELASTIC_REQUIRE set "TEST_ELASTIC_REQUIRE=1"
rem test-mmvq-planes-fork too on the vulkan machine when it ran no int8 mat-vec (the cuda build has no vulkan device)
if /i "%BACKEND%"=="vulkan" if not defined TEST_MMVQ_PLANES_REQUIRE set "TEST_MMVQ_PLANES_REQUIRE=1"
rem and its cases of the draft checks on the f16 coopmat tile of mul_mm_f16reg.comp (GGML_VK_MM_F16REG_VERIFY, 13.11)
if /i "%BACKEND%"=="vulkan" if not defined TEST_MMVQ_VERIFY set "TEST_MMVQ_VERIFY=1"
if not defined WARMUP set "WARMUP=0"
if not defined GGML_ALLOC_PEAK set "GGML_ALLOC_PEAK=1"
rem the vocabulary subset of the MTP head is on by default since 10.11 and takes its base from a counts file in the
rem llama.cpp cache that every server adds to: the runs keep the base of a new install (the first ids) and write
rem nothing, so a result does not depend on what the machine served before. an arm that sets LLAMA_MTP_VOCAB_STATS
rem (06: history) measures a frequency base
if not defined LLAMA_MTP_VOCAB_STATS set "LLAMA_MTP_VOCAB_STATS=none"
if not defined LLAMA_MTP_VOCAB_STATS_SAVE set "LLAMA_MTP_VOCAB_STATS_SAVE=0"
rem the upstream of 07 and of 00-build.bat upstream: the master of the last merge into cache, not the local branch
rem master, which a test machine does not pull (12.11 the 395 had df1170213 of 20.10 there, and 07 started on its own
rem refused the build of 0d69255c5). after a merge of master set it to that commit
if not defined UPSTREAMREF set "UPSTREAMREF=0d69255c5"

set "BIN=%BUILD%\bin\Release"
if not exist "%BIN%" set "BIN=%BUILD%\bin"
set "LOGS=%~dp0logs"
if not exist "%LOGS%" mkdir "%LOGS%"
set "PY="
where py >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if defined PY goto :eof
where python >nul 2>&1
if not errorlevel 1 set "PY=python"
