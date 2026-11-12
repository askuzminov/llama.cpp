@echo off
rem copy this to _local.bat and keep only the lines you change. _local.bat is ignored by git, so pulling never
rem touches it. _config.bat calls it after its first block; README.md describes every value.

rem ---- needed: the model and its MTP head ----
rem 395:
rem set "MODEL=C:\Users\AI\Downloads\lm-studio-models\unsloth\Qwen3.8-Flash-Next-GGUF\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
rem set "SPECDRAFT=C:\Users\AI\Downloads\lm-studio-models\unsloth\Qwen3.8-Flash-Next-GGUF\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"
rem 3090:
rem set "MODEL=E:\ml\lm-studio-models\unsloth\Qwen3.8-Flash-Next-GGUF\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
rem set "SPECDRAFT=E:\ml\lm-studio-models\unsloth\Qwen3.8-Flash-Next-GGUF\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"

rem ---- the 3090: the layout of the probe runs (48 by default, as the server; about 1.5 GiB a layer on the card) ----
rem set "EXTRA=-ngl 99 -ncmoe 46"

rem ---- a long natural text for 02-long: at least the native context plus one token (about 1.1 M characters) ----
rem set "QA_CORPUS=C:\data\long-text.txt"

rem ---- a switch measured against the default in the probes of 01 and 02 ----
rem set "QA_QARMS=default;pool32 LLAMA_QSA_POOL_F32=1"
rem set "QA_PARMS=default;nolist LLAMA_QSA_FA_LIST=0"
rem run-all.bat: a step out (RUN_BUILD, RUN_UPSTREAM, RUN_MICRO, RUN_CHECK, RUN_DECODE, RUN_PROFILE, RUN_KLD,
rem RUN_EXACT, RUN_LONG), the variants of the probes of 02-long and of its four conversations at once
rem set "RUN_LONG=0"
rem set "LONG_QARMS=default"
rem set "QA_PAR_ARMS=default"

rem ---- only some parts of 02 ----
rem set "QA_LONG_STAGES=chat par"

rem ---- build: pin the backend or the generator when the guess is wrong ----
rem set "CMAKE_BACKEND=-DGGML_CUDA=ON"
rem set "GENERATOR=Ninja"
