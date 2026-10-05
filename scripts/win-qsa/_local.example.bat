@echo off
rem copy this to _local.bat and keep only the lines you change. _local.bat is ignored by git, so
rem pulling never touches it. _config.bat calls it after its defaults; the lists that differ between
rem the machines come after it and keep what it sets. README.md describes every value.
rem
rem an older _local.bat: since 05.10.2026 the scripts read only the names in this file and in
rem README.md. the variables of the old scripts do nothing now, delete them: SPECAUTO*, SPECARMS,
rem SPECNGEN, SPECPROMPTFILE*, MOE*, UBMEM*, AB*, FAVARIANTS, REPS, DEPTHS, PERF*, TILES, INT8*,
rem DEPTH32-34, ARMS32-34, SRVCTX, SRVARGS. keep MODEL, SPECDRAFT, and EXTRA on the 3090. QCTX and
rem QCHUNKS keep their names and now set 03-quality: delete an old QCTX=65536 there, 03 needs about
rem 12 GB of ram at its default 16384 and fails at 32768 on the 395.
rem
rem the first run on a machine measures everything, there are no kept results yet. the arm
rem "stable @stable" needs 00-build.bat stable once (a build-stable of the old 35-build-stable.bat
rem works as it is).

rem ---- needed: the model and the MTP draft ----
rem 395:
rem set "MODEL=C:\Users\AI\Downloads\lm-studio-models\unsloth\Qwen3.8-Flash-Next-GGUF\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
rem set "SPECDRAFT=C:\Users\AI\Downloads\lm-studio-models\unsloth\Qwen3.8-Flash-Next-GGUF\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"
rem 3090:
rem set "MODEL=E:\ml\lm-studio-models\unsloth\Qwen3.8-Flash-Next-GGUF\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
rem set "SPECDRAFT=E:\ml\lm-studio-models\unsloth\Qwen3.8-Flash-Next-GGUF\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"

rem ---- needed on the 3090: -ncmoe that fits the card (the default is 30) ----
rem the static layout of 02, 03 and 05: llama-bench and llama-perplexity have no expert cache. 04-decode
rem adds DECARGS after it, there -ncmoe 48 --moe-cache auto wins: all 48 MoE layers on the cpu, the
rem free VRAM goes to the cache slots
rem set "EXTRA=-ngl 99 -ncmoe 46"

rem ---- a comparison of a switch: add an arm, the kept arms do not run again ----
rem set "BENCHARMS=stable @stable;default;radix0 GGML_VK_TOPK_RADIX=0"
rem set "BENCHDEPTHS=0,65536,122880"
rem set "QARMS=default;radix0 GGML_VK_TOPK_RADIX=0"
rem set "DECARMS=none;n3 @draft --spec-type draft-mtp --spec-draft-n-max 3"

rem ---- decode after a long prompt: 131072 chars of PPLFILE in front of each prompt ----
rem set "DECPREFIX=0 131072"
rem set "DECPROMPTS=os code"
rem set "DECSEEDS=1"

rem ---- the same answers with and without a switch: greedy sampling and pairs of arms ----
rem set "DECSAMP=0 20 0.95"
rem set "DECSAME=base:overlap n2:overlap-n2"

rem ---- the warmup run at start back on (off by default: --no-warmup) ----
rem set "WARMUP=1"

rem ---- per-op profile of a decode step as well ----
rem set "PROFKINDS=pp tg"

rem ---- build: pin the backend or the generator when the guess is wrong ----
rem set "CMAKE_BACKEND=-DGGML_CUDA=ON"
rem set "GENERATOR=Ninja"

rem ---- run-all.bat: steps this machine does not need ----
rem set "RUN_QUALITY=0"

rem ---- source of the wikitext text, if huggingface is unreachable ----
rem set "PPLURL=http://.../wikitext-2-raw-v1.zip"
