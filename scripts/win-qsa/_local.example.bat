@echo off
rem copy this to _local.bat and keep only the lines you actually change.
rem _local.bat is ignored by git, so pulling never touches it.
rem it is called from _config.bat after the defaults and before BIN/LOGS
rem are derived, so overriding BUILD here works too.

rem set "MODEL=D:\models\qwen3-next-Q4_K_M.gguf"

rem the backend is detected from what is installed: nvcc or CUDA_PATH -> cuda, else the
rem vulkan sdk -> vulkan. EXTRA follows it. set them here only to overrule that, or to
rem give -ncmoe a value that fits this card

rem strix halo / vulkan
rem set "CMAKE_BACKEND=-DGGML_VULKAN=ON"
rem set "EXTRA=-ngl 99"

rem 3090 + i9, moe experts on the cpu
rem set "CMAKE_BACKEND=-DGGML_CUDA=ON"
rem set "EXTRA=-ngl 99 -ncmoe 30"

rem the cmake generator. auto keeps the cmake default and falls back to ninja when cuda is
rem built without its msbuild integration. pin it when that guess is wrong
rem set "GENERATOR=Ninja"

rem source of the wikitext text for 05 and 06, if huggingface is unreachable
rem set "PPLURL=http://.../wikitext-2-raw-v1.zip"

rem arms of 04, 06 and 07: 1 = gather path, 0 = dense kernel. vulkan knobs, so off vulkan
rem they all measure the same thing and only arm 1 runs
rem set "FAVARIANTS=1 0"
rem set "PERFDEPTH=122880"
rem set "PERFPP=2048"
rem set "PERFUB=1024"
rem set "TILES="512,128,128,32,64,32,2,16,16,16,64" "256,128,64,32,64,32,2,16,16,16,64""
rem set "INT8CTX=4096"
rem set "INT8CHUNKS=2"
rem 32-depth.bat: the older build to compare with, and the depths of the sweep
rem set "OLDBIN32=C:\Users\AI\Documents\github\llamacpp\llama.cpp\build-win\bin\Release"
rem set "DEPTH32=0,65536,122880"

rem 33-indexer.bat: a shorter depth, and one arm with all three switches back at once
rem set "DEPTH33=65536"
rem set "ARMS33="default" "old LLAMA_INPUT_THREADS=1 GGML_VK_DISABLE_MM_HEADSUM=1 GGML_VK_DISABLE_MM_RELU=1""
rem context and chunks of the quality runs, keep well above the ~513 block budget
rem set "QCTX=65536"
rem set "QCHUNKS=2"

rem context and arguments of 14-server.bat. --cache-spill-dir keeps cold prompt caches
rem on disk, so they survive a restart
rem set "SRVCTX=262144"
rem set "SRVARGS=--host 0.0.0.0 --port 8080 --jinja --cache-spill-dir D:\llama-cache"

rem repeats per measurement in 03, 04 and 11, only worth raising if two runs disagree
rem set "REPS=2"

rem run-all runs every step by default, turn off what this machine does not need
rem set "RUN_BUILD=0"
rem set "RUN_KL=0"
rem set "RUN_DIAG=0"
rem set "RUN_PLE=0"

rem how many wikitext chunks 12-ple-real.bat runs
rem set "PLECHUNKS=8"

rem 17-spec-np.bat: the MTP draft gguf, and which arms to run. without SPECDRAFT the step
rem refuses to start. point SPECPROMPTFILE1 and SPECPROMPTFILE2 at your own prompts to use
rem those instead of the built-in pair; the two must differ or np2-free repeats np2-lock
rem set "SPECDRAFT=C:\models\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
rem set "SPECARMS=np2-lock np2-free"
rem set "SPECNGEN=1024"
rem set "SPECPROMPTFILE1=%~dp0prompt-a.txt"
rem set "SPECPROMPTFILE2=%~dp0prompt-b.txt"
rem set "RUN_SPEC=1"

rem diskspd for 13-disk-iops.bat: path to the exe if it cannot be downloaded here
rem set "DISKSPD=C:\tools\diskspd\amd64\diskspd.exe"
rem set "DISKVARIANTS="4 32 4" "256 32 4""
rem set "RUN_DISK=0"

rem 20-moecache.bat: arms "ctx np [args]", tokens generated per arm, and the prompt text. a saved
rem agent session in MOEPROMPTFILE measures the routing of that kind of text. the -ub of models.ini
rem goes into MOEARGS
rem set "MOEARMS="262144 1" "262144 1 -ncmoe 48 --moe-cache auto" "131072 2 -ncmoe 48 --moe-cache auto""
rem set "MOENGEN=4096"
rem set "MOEPROMPTFILE=%~dp0agent-session.txt"
rem set "MOEARGS=-fit off -ub 2048"
rem set "RUN_MOECACHE=1"
rem replay arms "ub [args]". auto at ub 1 takes more slots than at the -ub of the server, a
rem number pins the cache to the slots of the server: see its moe_cache line in the server log
rem set "MOEREPLAY="1" "1 -ncmoe 48 --moe-cache 37" "4 -ncmoe 48 --moe-cache 37""
rem spec arms "types [args]" run when SPECDRAFT of 17 above is set, none skips them
rem set "MOESPEC=none"
rem set "MOESPEC="draft-mtp -ncmoe 48 --moe-cache auto --temp 0""
rem set "MOEDMON=0"

rem 22-ubmem.bat: static arms and --phase-mem arms, "ub" or "ub ctx", none skips one kind, and the
rem server setup of models.ini. a smaller ctx makes the buffers that grow with n_ctx x n_ubatch
rem smaller. a larger UBMEMFILL makes the long prompt deeper: FILL x largest ub, below the ctx
rem set "UBMEMSTATIC=2048"
rem set "UBMEMARMS=8192 16384 "16384 131072""
rem set "UBMEMFILL=12"
rem set "UBMEMARGS=-fit off -ncmoe 48 --moe-cache auto"
rem set "UBMEMDRAFT=0"
rem set "RUN_UBMEM=1"

rem 35-37: stable baseline, long-context quality, PP and TG.
rem set "ABBIN=D:\llama-stable-build\bin\Release"
rem set "ABDEPTHS=0,65536,139264"
rem set "ABQCTX=32768"
rem set "ABQCHUNKS=1"
rem set "ABKEEPLOGITS=1"
