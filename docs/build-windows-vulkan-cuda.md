# Сборка на Windows: Vulkan, CUDA, ROCm

Заметка под две целевые машины этого форка:

- AMD AI Max 395 (Strix Halo, 128 GB) - **Vulkan** или **ROCm/HIP**, всё считается на iGPU;
- i9-12900 + RTX 3090 (128 GB) - **CUDA**.

Общий раздел [docs/build.md](build.md) остаётся источником истины по опциям;
здесь только то, что нужно для этих конфигураций, без лишних веток.

## 1. Что поставить

Общее для обеих машин:

- **Visual Studio 2022 Build Tools** (или Community). В инсталляторе нужны:
  - Workload: `Desktop development with C++`;
  - Components: `C++ CMake Tools for Windows`, `MSBuild support for LLVM (clang-cl) toolset`
    (последнее - только если будете собирать пресетами `x64-windows-llvm-*`).
- **Ninja** - ставится вместе с `C++ CMake Tools`, отдельно не нужен.
- **Git for Windows**.

Дальше по бэкендам:

- Vulkan: **LunarG Vulkan SDK** <https://vulkan.lunarg.com/sdk/home#windows>,
  установка по умолчанию. SDK даёт `glslc.exe` (обязателен, шейдеры компилируются
  на этапе сборки), заголовки Vulkan и SPIRV-Headers.
- CUDA: **CUDA Toolkit 12.x** <https://developer.nvidia.com/cuda-downloads>.
  Ставить **после** Visual Studio, иначе инсталлятор не пропишет интеграцию с MSBuild.
- ROCm: **AMD HIP SDK for Windows 10.x** <https://rocm.docs.amd.com/projects/install-on-windows/>.
  Ставить тоже после Visual Studio. Инсталлятор выставляет `HIP_PATH`; проверьте,
  что переменная есть в окружении, дальше всё пляшет от неё.

Все команды ниже запускаются из **x64 Native Tools Command Prompt for VS 2022**
(или из PowerShell после `Import-VisualStudioVars`). Обычный `cmd` не подойдёт:
не будет `cl.exe` в `PATH`.

Проверка, что окружение поднялось:

```
cl
cmake --version
ninja --version
glslc --version      :: только для Vulkan
nvcc --version       :: только для CUDA
echo %HIP_PATH%      :: только для ROCm
"%HIP_PATH%\bin\clang++" --version
```

## 2. Сборка с Vulkan (Strix Halo)

В репозитории есть готовый пресет:

```
cmake --preset x64-windows-vulkan-release
cmake --build build-x64-windows-vulkan-release -j
```

Пресет разворачивается в `-G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release`,
каталог сборки - `build-x64-windows-vulkan-release`.

Если нужен явный вариант без пресета:

```
cmake -B build-vk -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_VULKAN=ON ^
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build-vk -j
```

Замечания:

- Сборка шейдеров - самая долгая часть (несколько тысяч вариантов SPIR-V).
  Первый прогон занимает 10-20 минут, инкрементальные - секунды.
- Какие расширения GLSL доступны, определяется версией `glslc` из SDK.
  На этапе configure это видно в логе строками вида
  `GL_KHR_cooperative_matrix supported by glslc`. Для RDNA 3.5 важен
  `GL_KHR_cooperative_matrix` (coopmat1); `GL_NV_cooperative_matrix2` там
  всё равно недоступен, это расширение NVIDIA.
- `-DLLAMA_BUILD_TESTS=ON` нужен только если планируете гонять
  `test-backend-ops`, иначе выключайте: он заметно удлиняет сборку.

## 3. Сборка с CUDA (RTX 3090)

```
cmake -B build-cuda -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=86-real ^
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build-cuda -j
```

`CMAKE_CUDA_ARCHITECTURES=86-real` - это ровно RTX 3090 (Ampere, sm_86).
Без этого флага CMake соберёт набор `75-virtual 80-virtual 86-real 89-real 90-virtual`
(см. [ggml/src/ggml-cuda/CMakeLists.txt:26-56](../ggml/src/ggml-cuda/CMakeLists.txt#L26-L56)),
то есть в несколько раз дольше и в несколько раз больше по размеру.

Полезные опции CUDA:

- `-DGGML_CUDA_FA_ALL_QUANTS=ON` - flash attention для всех типов KV-кеша.
  По умолчанию `OFF`, собираются только распространённые комбинации.
  Нужно, если гоняете нестандартные `--cache-type-k/-v`. Сильно удлиняет сборку.
- `-DGGML_CUDA_FORCE_MMQ=ON` - только квантованные matmul-ядра вместо cuBLAS.
  Имеет смысл проверить на MoE-моделях, но по умолчанию не трогать.
- `-DGGML_CUDA_GRAPHS=ON` - включено по умолчанию, отключать не надо.

## 4. Сборка с ROCm/HIP (Strix Halo)

Альтернатива Vulkan на той же машине. Radeon 8060S в AI Max 395 - это **gfx1151**
(RDNA 3.5); форк её знает и подбирает отдельные конфигурации MMQ, см.
[ggml/src/ggml-cuda/vendors/hip.h:219-221](../ggml/src/ggml-cuda/vendors/hip.h#L219-L221)
и [ggml/src/ggml-cuda/common.cuh:83](../ggml/src/ggml-cuda/common.cuh#L83)
(в комментарии прямо назван AI Max 395).

```
set PATH=%HIP_PATH%\bin;%PATH%
cmake -B build-hip -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_HIP=ON ^
  -DGPU_TARGETS=gfx1151 ^
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ^
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build-hip -j
```

Почему именно так:

- Компилятор нужен из HIP SDK, а не из Visual Studio. `PATH` с `%HIP_PATH%\bin`
  впереди даёт `clang`/`clang++` от AMD.
- CMake на Windows не поддерживает язык HIP, поэтому сборка идёт по ветке
  `CXX_IS_HIPCC` в [ggml/src/ggml-hip/CMakeLists.txt:18-23](../ggml/src/ggml-hip/CMakeLists.txt#L18-L23),
  то есть через обычный C++-компилятор. Генератор обязан быть Ninja.
- `GPU_TARGETS` пробрасывается в `CMAKE_HIP_ARCHITECTURES`
  ([там же, строки 35-43](../ggml/src/ggml-hip/CMakeLists.txt#L35-L43)).
  Одна цель вместо списка - это заметно короче по времени сборки.
- Минимальная версия, которую пропускает CMake, - **HIP 6.1**:
  `if (${hip_VERSION} VERSION_LESS 6.1) message(FATAL_ERROR ...)`
  ([ggml/src/ggml-hip/CMakeLists.txt:54-56](../ggml/src/ggml-hip/CMakeLists.txt#L54-L56)).
  Верхняя граница не проверяется.

Опции HIP из [ggml/CMakeLists.txt:215-220](../ggml/CMakeLists.txt#L215-L220):

- `GGML_HIP_GRAPHS` - ON по умолчанию, не трогать;
- `GGML_HIP_NO_VMM` - ON по умолчанию;
- `GGML_HIP_MMQ_MFMA` - MFMA для CDNA, к gfx1151 отношения не имеет;
- `GGML_HIP_RCCL` - только для мульти-GPU, здесь не нужно;
- `GGML_HIP_EXPORT_METRICS` - метрики ядер, для разовой профилировки.

Что проверить перед первым запуском:

- В `%HIP_PATH%\bin\rocblas\library` должны быть файлы с `gfx1151` в имени.
  Если их нет, rocBLAS упадёт уже в рантайме, а не на сборке. Тогда либо ставить
  версию HIP SDK, в которой gfx1151 есть, либо собирать с Vulkan.
- `HSA_OVERRIDE_GFX_VERSION` на Windows **не работает**
  (см. [docs/build.md:397](build.md) и <https://github.com/ROCm/ROCm/issues/2654>),
  подменить архитектуру не выйдет.
- `llama-bench.exe --list-devices` покажет, сколько памяти HIP считает доступной.
  На Strix Halo это зависит от размера UMA-фреймбуфера в BIOS: если там выставлено
  мало, ROCm увидит мало, даже когда в системе 128 GB.

Что из Vulkan и ROCm быстрее на этой машине, заранее сказать нельзя: пути кода
разные (Vulkan идёт через coopmat1, HIP - через ядра ggml-cuda с ветками RDNA3_5).
Мерить `llama-bench` на своей модели, см. раздел 9.

## 5. Обе в одной сборке

Технически можно:

```
cmake -B build-all -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_VULKAN=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build-all -j
```

На Strix Halo аналогично можно собрать Vulkan и HIP вместе:

```
set PATH=%HIP_PATH%\bin;%PATH%
cmake -B build-all -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_VULKAN=ON -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 ^
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build-all -j
```

В любой такой сборке появляются **два** устройства, и слои по умолчанию
раскидываются по обоим. Выбирать нужное при запуске: `-dev CUDA0`, `-dev ROCm0`
или `-dev Vulkan0` (список - `--list-devices`,
[common/arg.cpp:2776-2790](../common/arg.cpp#L2776-L2790)).

`GGML_CUDA=ON` и `GGML_HIP=ON` одновременно смысла не имеют: обе опции собирают
одни и те же исходники `ggml/src/ggml-cuda/*.cu`, просто разными компиляторами.

Для повседневной работы проще держать отдельный каталог сборки под каждый бэкенд.

## 6. Проверка, что бэкенд поднялся

```
build-x64-windows-vulkan-release\bin\llama-bench.exe --list-devices
build-cuda\bin\llama-bench.exe --list-devices
build-hip\bin\llama-bench.exe --list-devices
```

Ожидаемые имена устройств:

| Сборка | Устройство |
| --- | --- |
| `GGML_VULKAN=ON` | `Vulkan0: AMD Radeon(TM) 8060S Graphics` |
| `GGML_CUDA=ON`   | `CUDA0: NVIDIA GeForce RTX 3090` |
| `GGML_HIP=ON`    | `ROCm0: AMD Radeon(TM) 8060S Graphics` |

Префикс задаётся `GGML_CUDA_NAME` в
[ggml/include/ggml-cuda.h:11-17](../ggml/include/ggml-cuda.h#L11-L17):
для HIP-сборки это именно `ROCm`, а не `HIP`.

Если Vulkan-устройство не появилось - проверьте, что стоит свежий драйвер AMD
(Adrenalin), а не Microsoft Basic Display Adapter, и что `vulkaninfo --summary`
из SDK видит физическое устройство.

Если не появилось `ROCm0` - проверьте `%HIP_PATH%\bin\amdhip64_*.dll` рядом с
бинарником или в `PATH`, и наличие gfx1151 в `%HIP_PATH%\bin\rocblas\library`.

## 7. Запуск

Флаги, которыми пользуются на этих машинах:

```
:: Strix Halo, Vulkan, всё на iGPU
llama-server.exe -m <model>.gguf ^
  -ngl 99 --flash-attn on ^
  --load-mode dio --lazy-mode on ^
  -c 131072 -ub 2048 -b 2048

:: Strix Halo, ROCm/HIP, всё на iGPU
llama-server.exe -m <model>.gguf ^
  -ngl 99 --flash-attn on ^
  --load-mode dio --lazy-mode on ^
  -c 131072 -ub 2048 -b 2048

:: 3090, CUDA
llama-server.exe -m <model>.gguf ^
  -ngl 99 --flash-attn on ^
  --load-mode dio --lazy-mode on ^
  -c 131072 -ub 2048 -b 2048
```

Набор флагов одинаковый для всех трёх бэкендов; отличается только сборка.

`--load-mode` и `--lazy-mode` описаны в
[common/arg.cpp:2728-2759](../common/arg.cpp#L2728-L2759).

## 8. Как на Windows работает `--load-mode dio`

На Windows флаг даёт две независимые вещи: небуферизованное чтение файла модели
и асинхронный конвейер загрузки. Раньше работала только вторая.

### Небуферизованное чтение

`llama_file::impl` открывает файл обычным `ggml_fopen`, а затем, если запрошен
`use_direct_io` и режим `"rb"`, переоткрывает тот же дескриптор через `ReOpenFile`
с `FILE_FLAG_NO_BUFFERING`
([src/llama-mmap.cpp:132-158](../src/llama-mmap.cpp#L132-L158)).
Буферизованный `FILE *` остаётся открытым: `file_id()` отдаёт его дескриптор
`llama_mmap`, когда тот же файл дополнительно отображается (`--lazy-mode on`).

Выравнивание берётся из `FILE_STORAGE_INFO`
(`LogicalBytesPerSector` / `PhysicalBytesPerSectorForAtomicity`), минимум 4096
([src/llama-mmap.cpp:116-130](../src/llama-mmap.cpp#L116-L130)). Если `ReOpenFile`
недоступен или файл лежит там, где `FILE_FLAG_NO_BUFFERING` не поддерживается,
пишется `LLAMA_LOG_WARN` и всё продолжает работать через буферизованный путь -
как на Linux при неудачном `O_DIRECT`.

Дальше логика повторяет POSIX-ветку:

- `read_raw` при активном direct I/O идёт через `read_aligned_chunk` -
  временный буфер от `_aligned_malloc`, чтение по границам сектора, `memcpy` в цель;
- `read_raw_unsafe` читает напрямую и предполагает, что вызывающий уже выровнял
  смещение, длину и адрес буфера (единственный такой вызов - конвейер в
  `load_all_data`);
- чтение за концом файла (добивка до границы сектора) не ошибка, хвост обнуляется;
- `has_direct_io()` теперь возвращает реальное состояние
  ([src/llama-mmap.cpp:270-272](../src/llama-mmap.cpp#L270-L272)), а не `true`
  безусловно.

Практическое следствие: `alignment` перестал быть равен 1, а от него зависит
размер порции в конвейере
([src/llama-model-loader.cpp:1513](../src/llama-model-loader.cpp#L1513)):

```cpp
const size_t buffer_size = alignment != 1 ? 64 * 1024 * 1024 + 2 * alignment : 1 * 1024 * 1024;
```

То есть порции выросли с **1 MB** до **64 MB**, и чтение больше не проходит через
кеш страниц Windows: нет лишней копии ядро -> pinned-буфер и нет вытеснения
полезного кеша объёмом модели.

### Асинхронный конвейер

Он включается не от direct I/O, а от того, что `dio` - единственный режим, который
одновременно не `mmap` и не `none`
([src/llama-model-loader.cpp:559-560](../src/llama-model-loader.cpp#L559-L560)):

```cpp
this->use_mmap      = load_mode == MMAP || load_mode == MMAP_MLOCK || load_mode == AUTO;
this->use_direct_io = load_mode == DIRECT_IO;
```

Лямбда `upload_backend` в `load_all_data` возвращает `nullptr`, если
`use_mmap || check_tensors`
([src/llama-model-loader.cpp:1519-1522](../src/llama-model-loader.cpp#L1519-L1522)).
Без mmap она отдаёт живой бэкенд, и тензоры грузятся так
([src/llama-model-loader.cpp:1678-1730](../src/llama-model-loader.cpp#L1678-L1730)):
4 закреплённых staging-буфера по кругу, `read_raw_unsafe` в буфер,
`ggml_backend_tensor_set_async` + `ggml_backend_event_record`, и следующая порция
читается с диска, пока предыдущая уходит по DMA в VRAM.

С mmap этого пути нет вообще: там `ggml_backend_tensor_alloc` поверх отображения
или синхронный `ggml_backend_tensor_set`, то есть перекладывание страниц через
page fault, по одному тензору за раз, без совмещения с DMA. Именно поэтому `dio`
обгонял mmap ещё до появления небуферизованного чтения.

`-lm none` даёт тот же конвейер, но уже без небуферизованного чтения и с порциями
по 1 MB.

### Что нужно, чтобы конвейер включился

`upload_backend` требует от устройства `async`, `host_buffer` и `events`
одновременно ([src/llama-model-loader.cpp:1546-1550](../src/llama-model-loader.cpp#L1546-L1550)).

- Vulkan: все три `true`
  ([ggml-vulkan.cpp:19066-19069](../ggml/src/ggml-vulkan/ggml-vulkan.cpp#L19066-L19069)).
- CUDA и ROCm: `async` и `events` `true`, `host_buffer` отключается переменной
  `GGML_CUDA_NO_PINNED`. Ставить её - значит выключить быструю загрузку.

Ещё два условия: `--check-tensors` тоже глушит этот путь, и тензор не должен
лежать в host-буфере (для слоёв на CPU идёт обычный синхронный `read_raw`,
теперь тоже выровненный).

### На Strix Halo mmap выключается и сам

Vulkan сообщает `mmap_support = !is_integrated_gpu`
([ggml-vulkan.cpp:19070](../ggml/src/ggml-vulkan/ggml-vulkan.cpp#L19070)),
а `load_tensors` при `load_mode == AUTO` обходит устройства и гасит mmap, если
хоть одно его не поддерживает
([src/llama-model.cpp:1417-1426](../src/llama-model.cpp#L1417-L1426), `llama_model_base::load_tensors`).
На iGPU конвейер поднимается и без флага; `-lm dio` добавляет к нему
небуферизованное чтение и порции по 64 MB.

Отдельно: `--lazy-mode on` с `-lm dio` совместим. `lazy_read::add` проверяет
`llama_mmap::SUPPORTED` (возможность платформы, на Windows истинна), а не
выбранный `load_mode`, и `init_mappings` создаёт отображение по условию
`use_mmap || lazy.any()`. То есть PLE-эмбеддинги остаются на диске за отображением
(через буферизованный дескриптор), а остальные веса идут через асинхронный
конвейер выше.

## 9. Диагностика производительности

Vulkan (переменные окружения, флаги сборки не нужны -
[ggml-vulkan.cpp:7772-7783](../ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7772-L7783)):

```
set GGML_VK_PERF_LOGGER=1
set GGML_VK_PERF_LOGGER_FREQUENCY=1000
llama-bench.exe -m <model>.gguf -p 2048 -n 0
```

Даёт разбивку по операциям с временем и GFLOPS/s. На длинном контексте смотреть
в первую очередь на `FLASH_ATTN_EXT`, `TOP_K`, `MUL_MAT_ID`, `ADD`, `FILL`.

Другие переменные Vulkan, полезные при отладке регрессий:
`GGML_VK_DISABLE_COOPMAT`, `GGML_VK_DISABLE_COOPMAT2`, `GGML_VK_DISABLE_FUSION`,
`GGML_VK_DISABLE_GRAPH_OPTIMIZE`, `GGML_VK_FORCE_MAX_ALLOCATION_SIZE`.

CUDA и ROCm (одни и те же переменные, HIP-сборка использует тот же код
`ggml/src/ggml-cuda`):

```
set GGML_CUDA_DISABLE_FUSION=1
set GGML_CUDA_DISABLE_GRAPHS=1
set GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
```

Первые две - для локализации регрессий, `ENABLE_UNIFIED_MEMORY` - чтобы вместо
падения при нехватке VRAM уходить в системную память. Выбор устройств на уровне
рантайма: `GGML_CUDA_DEVICES`, либо `HIP_VISIBLE_DEVICES` для ROCm.

`GGML_CUDA_NO_PINNED` в этот список не входит: она снимает `host_buffer` с
устройства и тем самым отключает асинхронную загрузку модели, см. раздел 8.

Отдельного per-op логгера, как `GGML_VK_PERF_LOGGER`, у CUDA/HIP нет; для
разбивки по ядрам собирайте с `-DGGML_HIP_EXPORT_METRICS=ON` или профилируйте
внешними средствами.

Замер prefill отдельно от decode:

```
llama-bench.exe -m <model>.gguf -p 512,2048 -n 0 -ub 512,2048 -fa 1
```

Проверка корректности бэкенда против CPU (нужна сборка с `-DLLAMA_BUILD_TESTS=ON`):

```
test-backend-ops.exe test -o FLASH_ATTN_EXT
test-backend-ops.exe test -o MUL_MAT_ID
```
