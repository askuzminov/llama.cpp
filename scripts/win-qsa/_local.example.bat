@echo off
rem copy this to _local.bat and keep only the lines you actually change.
rem _local.bat is ignored by git, so pulling never touches it.
rem it is called from _config.bat after the defaults and before BIN/LOGS
rem are derived, so overriding BUILD here works too.

rem set "MODEL=D:\models\qwen3-next-Q4_K_M.gguf"

rem strix halo / vulkan
rem set "CMAKE_BACKEND=-DGGML_VULKAN=ON"
rem set "EXTRA=-ngl 99"

rem 3090 + i9, moe experts on the cpu
rem set "CMAKE_BACKEND=-DGGML_CUDA=ON"
rem set "EXTRA=-ngl 99 -ncmoe 30"

rem source of the wikitext text for 05 and 06, if huggingface is unreachable
rem set "PPLURL=http://.../wikitext-2-raw-v1.zip"

rem arms of 04, 06 and 07: 1 = gather path, 0 = dense kernel
rem set "FAVARIANTS=1 0"
rem set "PERFDEPTH=122880"
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

rem PLE row cache variants for 12-ple-real.bat: block size in bytes, and how many
rem wikitext chunks it runs
rem set "PLEREALVARIANTS="0 4096""
rem set "PLECHUNKS=8"

rem diskspd for 13-disk-iops.bat: path to the exe if it cannot be downloaded here
rem set "DISKSPD=C:\tools\diskspd\amd64\diskspd.exe"
rem set "DISKVARIANTS="4 32 4" "256 32 4""
rem set "RUN_DISK=0"
