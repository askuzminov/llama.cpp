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
rem seconds of rest after a model run or a build before a timed run: the 395 runs warm (95 C in an hour of chats)
if not defined COOL      set "COOL=120"
if not defined CHECKOPS  set "CHECKOPS=MUL_MAT;MUL_MAT_ID;MUL_MAT_ID_FUSION;MUL_MAT_VEC_FUSION;MUL_MAT_RELU;MUL_MAT_HEADSUM;TOP_K;TOPK_QSA;LIGHTNING_INDEXER;LIGHTNING_INDEXER_UNIT;GET_ROWS;GATED_DELTA_NET;GATED_DELTA_NET_CACHE_FUSION;DSV4_HC_PRE;DSV4_HC_POST;HC_POST_GATE;REPEAT;FLASH_ATTN_EXT n_kv_max=[1-9];RMS_NORM;RMS_NORM_SCALE;RMS_NORM_MUL_SIGMOID_MUL;SSM_CONV;CONCAT_SSM_CONV;MUL_MAT_ID_REPEAT;GATED_DELTA_NET_LAZY;FLASH_ATTN_LIST;DSV4_HC_PRE_F16;DSV4_HC_POST_NORM_F16"
rem int8: the int8 coopmat path of the experts, off by default (f16reg), checked as well
if not defined CHECKARMS set "CHECKARMS=default;int8 GGML_VK_MMID_F16REG=0 GGML_VK_MMID_GLU_MIN_TOKENS=0 @o=MUL_MAT_ID,MUL_MAT_ID_REPEAT,MUL_MAT_ID_FUSION,MUL_MAT_VEC_FUSION"
goto :defaults_done

:cuda_defaults
rem the 3090 (cuda): the static layout of the probe runs (llama-quality-fork has no expert cache) keeps 30 MoE
rem layers on the cpu; the server (DECARGS) all 48 with the expert cache in the free VRAM
if not defined EXTRA     set "EXTRA=-ngl 99 -ncmoe 30"
if not defined DECARGS   set "DECARGS=-fit off -ncmoe 48 --moe-cache auto --phase-mem --no-repack -b 16384 -ub 8192"
if not defined CHECKOPS  set "CHECKOPS=MUL_MAT;MUL_MAT_ID;MUL_MAT_ID_FUSION;MUL_MAT_VEC_FUSION;MUL_MAT_RELU;MUL_MAT_HEADSUM;TOP_K;TOPK_QSA;LIGHTNING_INDEXER;LIGHTNING_INDEXER_UNIT;GET_ROWS;GATED_DELTA_NET;GATED_DELTA_NET_CACHE_FUSION;DSV4_HC_PRE;DSV4_HC_POST;HC_POST_GATE;REPEAT;FLASH_ATTN_EXT n_kv_max=[1-9];RMS_NORM;RMS_NORM_SCALE;RMS_NORM_MUL_SIGMOID_MUL;SSM_CONV;CONCAT_SSM_CONV;MUL_MAT_ID_REPEAT;GATED_DELTA_NET_LAZY"
if not defined GGML_OP_OFFLOAD_MIN_BATCH set "GGML_OP_OFFLOAD_MIN_BATCH=128"
if not defined LLAMA_PREFILL_BUDGET set "LLAMA_PREFILL_BUDGET=0"

:defaults_done
rem test-elastic-buffer-fork fails when it tested no device (a skip must not count as a pass)
if not defined TEST_ELASTIC_REQUIRE set "TEST_ELASTIC_REQUIRE=1"
if not defined WARMUP set "WARMUP=0"
if not defined GGML_ALLOC_PEAK set "GGML_ALLOC_PEAK=1"

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
