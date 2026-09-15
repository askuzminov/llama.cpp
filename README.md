# Чем этот форк отличается от апстрима

Форк [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), рабочая ветка `cache`.
Раздел ведётся вручную: каждое изменение относительно апстрима дописывается сюда.

Целевое железо: AMD AI Max 395 (Strix Halo, 128 GB, Vulkan, всё на GPU) и i9-12900 + RTX 3090
(128 GB, MoE на CPU). Целевые модели: Qwen3.8-Flash-Next (`qwen4exp`), Qwen3.5 122B.
Развёртывание на Windows, mmap не используется: `--load-mode dio --lazy-mode on`.

## Загрузка модели

- Выделение буферов бэкенда совмещено с чтением файла. `ggml_backend_alloc_ctx_tensors_from_buft_cb`
  ([ggml/include/ggml-alloc.h](ggml/include/ggml-alloc.h)) отдаёт диапазоны тензоров по мере
  готовности буферов, загрузчик читает их, пока выделяется следующий буфер.
- Direct I/O на Windows: `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, запросы по 16 MiB, один
  в полёте ([src/llama-mmap.cpp](src/llama-mmap.cpp)). В апстриме direct I/O есть только под POSIX,
  под Windows `-lm dio` молча сводится к обычному `fread`.
- Заголовок GGUF читается тем же хендлом, что и тензоры ([src/llama-model-loader.cpp](src/llama-model-loader.cpp)).
  В апстриме `gguf_init_from_file` открывает файл отдельно и читает его через page cache, так что
  часть файла попадает в кеш ещё до весов. Пока по файлу идёт буферизованное чтение,
  небуферизованное по нему же почти останавливается: diskspd `-b4K -o1 -t32 -Suw` даёт 489.9 MiB/s
  и 125408 IOPS, при параллельном буферизованном читателе - 10.7 MiB/s и 2731 IOPS, и сразу после
  его окончания снова 489.7 MiB/s. Проверка -
  [scripts/win-qsa/13-disk-iops.bat](scripts/win-qsa/13-disk-iops.bat). `-lm dio -lzm on` смешивает
  два режима: ленивый тензор требует mmap, и загрузчик предупреждает об этом. `-lzm dio` оставляет
  файл неотображённым.
- Таблица PLE (`per_layer_token_embd.weight`, 26.82 GiB у `qwen4exp`) может остаться на диске:
  `-lzm dio` собирает её построчно ([src/llama-row-cache.cpp](src/llama-row-cache.cpp)) - пул
  читателей и небольшой кеш блоков с часовым вытеснением перед файловым кешем системы. Один блок -
  одна строка: блок в 4 KiB отдавал 90 полезных байт и вычитывал 4.47 GiB там, где сами строки
  занимают 0.25 GiB. Чтение буферизованное, поэтому страничный кеш держит выданные строки в любой
  свободной памяти и переживает выход процесса. В апстриме у `-lzm` есть только `on`, который
  требует mmap. Настройки через `LLAMA_ROW_CACHE_*`, замеры:
  [scripts/win-qsa/README.md](scripts/win-qsa/README.md).
- Замер на Strix Halo, llama-perplexity, 16 чанков wikitext, тёплый страничный кеш: `-lzm on`
  25.57 s/pass, `-lzm dio` 25.98 s/pass, PPL 3.9400 в обоих. Небуферизованное чтение той же
  таблицы давало 38.57 s/pass: на этом файле оно не набирает глубины очереди совсем - 6997
  запросов/с и на одном читателе, и на 32, при том что diskspd тем же шаблоном даёт 125408 IOPS.
  После нескольких GB чтений по файлу тот же путь выдаёт 71734 запроса/с. Причина не найдена, на
  машине включён Defender.
- Замер на Strix Halo, Qwen3.8-Flash-Next Q4_K_XL, 76.23 GiB на GPU:

  | | чтение | загрузка целиком |
  |---|---|---|
  | буферизованное чтение, выделение последовательно (как в апстриме) | 35.4 s @ 2.15 GiB/s | ~46-47 s, оценка |
  | direct I/O 16 MiB + совмещение | 23.3 s @ 3.27 GiB/s | 24.6 s |

  Размер запроса и overlapped-хендл нужны оба: те же 16 MiB на синхронном хендле дают 2.66 GiB/s,
  overlapped при 4 MiB - 2.67, больше одного запроса в полёте только замедляет (2 -> 2.98, 8 -> 2.53).

## Кэш промпта на сервере

- Контекстные чекпойнты хранятся деревом ветвлений, а не линейной цепочкой: `--ctx-checkpoints-tree`
  ([common/checkpoint-tree.h](common/checkpoint-tree.h)). Агент, который пробует несколько
  продолжений от общего префикса, возвращается к любой ветке без пересчёта префикса.
- Дельта-чекпойнты: `llama_state_seq_get_delta_ext` и `llama_state_seq_apply_delta`
  ([include/llama.h](include/llama.h)) сохраняют и накатывают только ячейки после `base_pos`.
- Вытеснение по остатку свободной RAM: `-crr, --cache-ram-reserve N`. Проверяется при создании
  каждого чекпойнта, поэтому учитывает и память, занятую другими процессами.
- Сброс холодных состояний на диск вместо выбрасывания: `--cache-spill-dir PATH`, бюджет
  `--cache-disk N`. Файлы переживают перезапуск и подхватываются при совпадении модели и
  конфигурации KV, то есть кэш работает и на холодном старте.
- Тесты: `tests/test-checkpoint-*.cpp`, `tests/test-prompt-cache-spill.cpp`,
  `tools/server/tests/unit/test_cache_spill.py`, `tools/server/tests/unit/test_ctx_checkpoint_tree.py`.

## Архитектура `qwen4exp`

- QSA: выбор блоков кэша идёт на каждый токен отдельно, как в референсе
  ([src/models/qwen4exp.cpp](src/models/qwen4exp.cpp)). Общее ранжирование на группу токенов было
  добавлено и удалено: ядро flash attention выбрасывает тайл маски только когда он замаскирован
  для всех строк запроса, и общий выбор давал до 2.1x на префиле, но это другой выбор, и он стоил
  от 0.26% до 1.04% PPL. Замеры и причина удаления: [scripts/win-qsa/README.md](scripts/win-qsa/README.md).
- MTP-голова: тензоры `NEXTN_HC_HEAD_{NORM,DOWN,UP}` в загрузчике и конвертере
  ([conversion/qwen4exp.py](conversion/qwen4exp.py)).

## Vulkan

- Разреженный flash attention на префиле: апстримовский путь (`flash_attn_sparse_compact.comp`)
  строит точный список ячеек KV на строку маски, но включается только на декоде, потому что
  тайл из нескольких строк запроса один список разделить не может. Наше дополнение: если
  `KV >= 8 * n_kv_max`, тюнер пересобирает тайл в одну строку (`block_rows == 1`, скалярный путь),
  и тогда список работает и на префиле. Тайл в одну строку теряет матричное ядро, поэтому порог
  по глубине нужен. `GGML_VK_FA_SPARSE_DISABLE=1` выключает разреженный путь целиком.
  Наш прежний гатер по тайлу (список на группу строк, ёмкость с запасом, откат на плотный путь
  при переполнении) снят в пользу апстримовского: он давал префилу +34% при `-ub 512` и +107%
  при `-ub 2048`, но результат отличался от плотного, а апстримовский список точный.
- `GGML_VK_ALLOC_TIMING=1` печатает, куда уходит время создания буферов: create, allocate, map.
- Порог hoisting row ids в `mul_mat_id` берётся из `maxComputeSharedMemorySize` устройства вместо
  зашитых 256 экспертов.

## Прочее

- [docs/build-windows-vulkan-cuda.md](docs/build-windows-vulkan-cuda.md) - сборка под Windows.
- [scripts/win-qsa/](scripts/win-qsa/) - батники для замеров на целевой Windows-машине, свой README.
  Там же ведётся список отброшенных вариантов и опровергнутых гипотез с числами, чтобы не
  возвращаться к ним по второму разу.

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
