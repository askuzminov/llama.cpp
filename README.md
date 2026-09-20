# Чем этот форк отличается от апстрима

Форк [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), рабочая ветка `cache`.
Раздел ведётся вручную: каждое изменение относительно апстрима дописывается сюда.

Целевое железо: AMD AI Max 395 (Strix Halo, 128 GB, Vulkan, всё на GPU) и i9-12900 + RTX 3090
(128 GB, MoE на CPU). Целевые модели: Qwen3.8-Flash-Next (`qwen4exp`), Qwen3.5 122B.
Развёртывание на Windows, mmap не используется: `--load-mode dio`, режим `--lazy-mode` по
умолчанию `auto`, и без mmap он сам сводится к `dio`.

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
  читателей поверх буферизованного чтения, без mmap. Один блок - одна строка: блок в 4 KiB отдавал
  90 полезных байт и вычитывал 4.47 GiB там, где сами строки занимают 0.25 GiB. Своего кеша блоков
  нет: он ничего не давал (числа ниже), и перед файлом остаётся один страничный кеш системы. Чтение
  буферизованное, поэтому страничный кеш держит выданные строки в любой свободной памяти и
  переживает выход процесса. В апстриме у `-lzm` есть только `on`, который требует mmap; `auto` без
  mmap теперь сводится к `dio`, а не выключается. Настройки через `LLAMA_ROW_CACHE_*`, замеры:
  [scripts/win-qsa/README.md](scripts/win-qsa/README.md).
- Замер на Strix Halo, llama-perplexity, 16 чанков wikitext по 32768, тёплый страничный кеш:
  `-lzm on` 216.18 т/с префила, `-lzm dio` 342.75 т/с, PPL 3.9400 в обоих. Свой кеш блоков на это
  не влиял - выключенный пул давал 342.15 т/с, 1024 МиБ 342.60 т/с, при попаданиях 0.3% и 43.1%,
  потому что те же страницы держит файловый кеш системы. Поэтому кеш блоков и синтаксис `dio:MiB`
  убраны, вместе с `lazy_cache_mib` и `LLAMA_ROW_CACHE_MIB`. Небуферизованное чтение той же таблицы давало 38.57 s/pass: на этом файле оно не
  набирает глубины очереди совсем - 6997 запросов/с и на одном читателе, и на 32, при том что
  diskspd тем же шаблоном даёт 125408 IOPS. После нескольких GB чтений по файлу тот же путь выдаёт
  71734 запроса/с. Причина не найдена, на машине включён Defender.
- Замер на Strix Halo, Qwen3.8-Flash-Next Q4_K_XL, 76.23 GiB на GPU:

  | | чтение | загрузка целиком |
  |---|---|---|
  | буферизованное чтение, выделение последовательно (как в апстриме) | 35.4 s @ 2.15 GiB/s | ~46-47 s, оценка |
  | direct I/O 16 MiB + совмещение | 23.3 s @ 3.27 GiB/s | 24.6 s |

  Размер запроса и overlapped-хендл нужны оба: те же 16 MiB на синхронном хендле дают 2.66 GiB/s,
  overlapped при 4 MiB - 2.67, больше одного запроса в полёте только замедляет (2 -> 2.98, 8 -> 2.53).

## Кэш промпта на сервере

- Дельта-чекпойнты: `llama_state_seq_get_delta_ext` и `llama_state_seq_apply_delta`
  ([include/llama.h](include/llama.h)) сохраняют и накатывают только ячейки после `base_pos`.
- Вытеснение по остатку свободной RAM: `-crr, --cache-ram-reserve N`. Проверяется при создании
  каждого чекпойнта, поэтому учитывает и память, занятую другими процессами.
- Сброс холодных состояний на диск вместо выбрасывания: `--cache-spill-dir PATH`, бюджет
  `--cache-disk N`. Файлы переживают перезапуск и подхватываются при совпадении модели и
  конфигурации KV, то есть кэш работает и на холодном старте.
- Буферы состояния выделяются выровненными на 4096 (`server_state_alloc` в
  [tools/server/server-task.h](tools/server/server-task.h)), поэтому спилл отдаёт их диску напрямую,
  без промежуточного `memcpy`, запросами по 16 МиБ - как загрузчик модели при `--load-mode dio`.
  Через буфер идут только заголовок, список токенов и хвост короче блока. На NVMe с обходом кэша ОС
  на 512 МиБ полезной нагрузки: запись 3.15 против 2.10 ГБ/с, чтение 3.10 против 1.86 ГБ/с. Формат
  файла не изменился - те же байты, что писала прежняя версия.
- При старте читаются только заголовок и список токенов каждого файла, полезная нагрузка - лениво,
  на первом попадании в кэш. Файл после чтения остаётся на месте.
- При штатной остановке содержимое слотов уходит в кэш до `destroy()`
  (`save_slots_to_cache` в [tools/server/server-context.cpp](tools/server/server-context.cpp)).
  Иначе состояние попадало в кэш только при запуске следующей задачи, и сервер, остановленный
  после последнего запроса, терял живой промпт: папка спилла оставалась пустой. То же самое перед
  уходом в сон.
- Запись с опережением: пока все слоты простаивают, содержимое слотов уходит в кэш, а резидентные
  состояния копируются на диск (`write_behind` в [tools/server/server-task.cpp](tools/server/server-task.cpp),
  вызов в ветке `all_idle` в [tools/server/server-context.cpp](tools/server/server-context.cpp)).
  Спилл перестал быть переносом: у состояния появилось третье положение - копия на диске при живых
  данных в RAM (`is_clean`). Что это даёт: вытеснение по `free_ram` больше не пишет ничего, оно
  просто освобождает RAM; попадание в кэш не ждёт чтения; `persist()` на остановке пишет только то,
  что не успело уйти; падение или `kill -9` больше не уносит весь кэш. Запись идёт от старых к
  новым - именно старые вытесняются первыми. Очередь задач (`server_queue::has_new_task`) проверяется не
  только между состояниями, но и внутри записи, каждые 16 МиБ - иначе запрос ждал бы, пока
  допишется многогигабайтный блоб (на 5 ГБ и 3 ГБ/с это 1.7 с). Прерванная запись удаляет свой
  файл, состояние остаётся грязным и уходит в следующий простой.
- По таймеру не пишем. На `qwen4exp` таблица PLE стримится с диска во время генерации (из лога:
  874536 запросов по 180 Б, 8.02 с ожидания чтений, в среднем 20.9 в полёте, пик 32, 192 мкс на
  запрос), и последовательная запись по 16 МиБ добавила бы задержку этим мелким случайным чтениям.
  Поэтому только простой.
- Попадание в кэш не удаляет файл. Раньше удаляло - спилл был переносом, и запись жила на диске
  ровно до первого чтения. Для агента это означало полную перезапись состояния на каждый ход:
  попадание уносило файл, ход добавлял несколько сотен токенов, следующий простой писал те же
  гигабайты заново. Теперь `unspill_state` только читает, а запись остаётся в `states` как
  disk-only: бюджет её по-прежнему видит, ветка от этого же префикса по-прежнему восстановима, и
  никто не платит за повторную запись тех же байтов. Единственное место, которое удаляет файлы по
  собственной инициативе - `enforce_disk_limit`.
- Шаг перезаписи. Если начало состояния уже лежит на диске, `write_behind` не пишет его заново,
  пока оно не выросло на `--cache-min-tokens` сверх этого префикса: перезапись покупает только
  хвост, а стоит она целого блоба. Падение при этом обходится максимум в `N` токенов пересчёта
  (при 8192 и 165.8 т/с - 49 с), а не в весь промпт. При `N = 0` порога нет и поведение прежнее.
  После успешной записи более длинного состояния его disk-only префиксы удаляются: это
  дедупликация, единственная копия чего бы то ни было при этом не трогается.
  Соответственно `alloc()` больше не выбрасывает вложенный префикс, если тот на диске, - у него
  забирается только RAM.
- `--cache-min-tokens N` тем самым задаёт одно: минимальный объём работы, ради которого стоит
  трогать диск. Промпты короче `N` в кэш не попадают вовсе - короткий промпт дёшево пересчитать, а
  рекуррентная часть гибридного состояния одинакова при любой длине
  (`llama_memory_recurrent::state_write_data` пишет `cell_count` строк на слой независимо от длины
  промпта), поэтому короткие промпты занимают место, ничего не экономя. По умолчанию 0.
- Бюджет диска (`--cache-disk`) при переполнении снимает копию с резидентного состояния, а не
  выбрасывает запись целиком: данные в RAM остаются рабочими. Порядок удаления - по ценности
  `(n_used + 1) * n_tokens`: и редко используемое, и дешёвое в пересчёте уходит раньше длинного,
  за которое дорого платить повторно. Раньше учитывалось только `n_used`, и 200k-токенное
  состояние, ни разу не переиспользованное, уходило наравне с тысячетокенным.
- Токенный лимит кэша (`update()`) при заданном `--cache-disk` считает только резидентные токены.
  Иначе два ограничителя дрались бы: FIFO по токенам выбрасывал бы disk-only записи по возрасту
  раньше, чем бюджет диска успел бы отсортировать их по ценности. Без `--cache-disk` токенный
  лимит остаётся единственной границей папки и продолжает их считать.
- Тесты: `tests/test-checkpoint-*.cpp`, `tests/test-prompt-cache-spill.cpp`,
  `tools/server/tests/unit/test_cache_spill.py`.

## Архитектура `qwen4exp`

- QSA: выбор блоков кэша идёт на каждый токен отдельно, как в референсе
  ([src/models/qwen4exp.cpp](src/models/qwen4exp.cpp)). Общее ранжирование на группу токенов было
  добавлено и удалено: ядро flash attention выбрасывает тайл маски только когда он замаскирован
  для всех строк запроса, и общий выбор давал до 2.1x на префиле, но это другой выбор, и он стоил
  от 0.26% до 1.04% PPL. Замеры и причина удаления: [scripts/win-qsa/README.md](scripts/win-qsa/README.md).
- Compute-буфер на полном контексте: `llama-server -c 262144 -ub 2048` просил у Vulkan 19.13 ГиБ и
  падал. Маска внимания лежит в CPU-буфере, а каждый разреженный слой строил от неё свой вид, и
  планировщик копировал её на бэкенд по разу на слой - 13 копий по 1 ГиБ. Плюс `relu` стоял после
  `reshape`, а переиспользовать буфер родителя аллокатор умеет только для реально размещённого
  родителя, не для вида, так что индексатор брал второй буфер на 2 ГиБ. Теперь вид на маску один на
  все слои (`graph::qsa_kq_mask_rows`), `ggml_fill` берёт его же, а `relu` стоит до `reshape`.
  Резерв на `-c 262144`: 5725 МиБ вместо 19581 при `-ub 2048`, 1400 вместо 4904 при `-ub 512`;
  на `-c 67584 -ub 512` - 490 вместо 1312. Значения не меняются.
- MTP-голова: тензоры `NEXTN_HC_HEAD_{NORM,DOWN,UP}` в загрузчике и конвертере
  ([conversion/qwen4exp.py](conversion/qwen4exp.py)).
- Общая MTP-голова (`mtp-...-shared-*.gguf` у unsloth, `qwen4exp.nextn_shared_target_tensors`) не
  несёт ни `token_embd.weight`, ни `output.weight`, и загрузчик падал на
  `check_tensor_dims: tensor 'token_embd.weight' not found`. Теперь у сайдкара без ствола оба
  тензора необязательны, а граф головы берёт их из модели, для которой он черновик, через уже
  существующий `cparams.ctx_other` - тот же механизм, что у `eagle3` и `dflash`
  ([src/models/qwen4exp.cpp](src/models/qwen4exp.cpp), [src/llama-context.cpp](src/llama-context.cpp)).
  Голова весит 2.60 ГБ вместо 3.85 ГБ у самодостаточной. Работает только с `-md`: без целевого
  контекста создание контекста головы отклоняется. Признак общей памяти в драйвере MTP теперь
  дополнительно требует, чтобы у черновика не было ствола, иначе `ctx_other` у заимствующей головы
  читался бы как общий KV ([common/speculative.cpp](common/speculative.cpp)).
- Проекция `hc_*_inject` считается с переставленными операндами. Она даёт всего `hc = 4` строки
  на входе шириной `hc * n_embd = 10240`, поэтому тайл матмула заполнен на 4 строки из 32, и узел
  стоил 753 мкс при 95 вызовах на граф (55.7 GFLOPS) - больше, чем соседняя `hc_*_down` на тех же
  данных с 320 строками (453 мкс). Переставленный `ggml_mul_mat(xn, w_inject)` даёт `[n_tokens, hc]`
  и попадает на ядро мат-вектора (`n <= 8` столбцов), обратный `ggml_cont(ggml_transpose(...))`
  стоит несколько КиБ. На Strix Halo узел стал 144 мкс вместо 753, транспонирование добавило
  3.8 мкс на вызов, граф префила - 2.058 с вместо 2.117 с, `pp512 @ d64000` - 236.29 т/с вместо
  228.01, PPL не изменился (4.2417 против 4.2449 при `+/- 0.036`). Лоры для этого тензора больше
  нет, веса hyper-connections не бывают целью адаптера.
  Перестановка кладёт вес в слот `src1`, а бэкенды читают `src1` только как f32, f16 или q8_1
  (`ggml_vk_get_dequantize_mul_mat_vec`, такой же запрет в `ggml_compute_forward_mul_mat` на CPU). В основной
  модели гаммы `hc_*_inject` - F32 [10240, 4], а в MTP-голове - Q8_0, поэтому сервер с `-md` падал на
  `ggml-vulkan.cpp:8113`. Теперь перестановка включается только для F32-гаммы, квантованная идёт обычным
  порядком `ggml_mul_mat(w_inject, xn)`, оба варианта дают `[hc, n_tokens]`.

## Спекулятивное декодирование (MTP)

Рабочие настройки для Qwen3.8-Flash-Next с общей MTP-головой:
`--spec-draft-n-min 0 --spec-draft-n-max 6 --spec-draft-p-min 0.6`.

- `n-min` держать на 0. При срабатывании нижней границы драйвер делает `dp.result->clear()`
  ([common/speculative.cpp](common/speculative.cpp)), то есть выбрасывает уже оплаченные декоды
  головы.
- `n-max` упирается не в отдачу драфта, а в порог flash attention. Проход целевой модели проверяет
  `1 + длина драфта` позиций, и пока это число не больше 8, Vulkan сворачивает GQA: `N` заменяется
  на `qk_ratio`, а `workgroups_y` делится на него, поэтому K/V читается один раз на kv-голову
  ([ggml/src/ggml-vulkan/ggml-vulkan.cpp](ggml/src/ggml-vulkan/ggml-vulkan.cpp), условие `N <= 8`).
  Дальше свёртка отключается, число workgroups растёт в `qk_ratio` раз, а тайл coopmat1 на 16 строк
  всё равно заполнен на 9. Лишняя стоимость прохода по числу проверяемых позиций: 4-11 мс при 7,
  24-26 мс при 8, 85-87 мс при 9, против базовых 7.39 мс на позицию.
- `n-max` стоит и памяти: `need_n_rs_seq()` отдаёт его в `cparams.n_rs_seq`
  ([common/common.h](common/common.h)), а буфер рекуррентных состояний берёт
  `mem_size * (1 + n_rs_seq)` строк ([src/llama-memory-recurrent.cpp](src/llama-memory-recurrent.cpp)).
  На Strix Halo при `-np 2` строка стоит 112.57 МиБ, то есть `n-max 6` - это 1576 МиБ против 1351 у
  `n-max 5`.
- `n-max` 5 и 6 по скорости неразличимы: усечением шести прогонов они расходятся меньше чем на 1%
  и меняются местами со счётом 3:3 (5 против 6: 36.03/36.09, 32.47/32.48, 32.98/32.71,
  31.02/30.71, 33.03/33.25, 34.37/34.32). Выбран 6. Разница между ними - 225 МиБ буфера
  рекуррентных состояний при `-np 2` и то, что 6 доходит до 7 проверяемых позиций, где начинаются
  лишние 4-11 мс на проход.
- Сравнивать настройки по разным прогонам нельзя: одна и та же 6/0.6 померена дважды и дала 36.08 и
  33.57 t/s. Сэмплирование не жадное, каждый прогон пишет свой текст. Мерить только усечением
  одного прогона по его же `acc per pos`, либо ставить `--temp 0`.
- Пересборка графа целевой модели стоит 4.32 мс. При `-np 1` переиспользуется 47 проходов из 206,
  оставшиеся 159 - это 687 мс из 15.74 с, 4.4%; полное переиспользование дало бы 33.94 t/s против
  32.46. Условие переиспользования требует равного `ubatch.n_tokens`
  ([src/llama-graph.h](src/llama-graph.h)), а `1 + длина драфта` меняется от прохода к проходу.
  Цена меряется без правок кода: `LLAMA_GRAPH_REUSE_DISABLE=1` даёт 0% переиспользования и
  32.04 t/s на той же последовательности токенов.
- `-np 2` окупается слабо: по стенке 30.40 -> 35.43 t/s на двух запросах, +16.5%, при этом каждый
  запрос получает 59% одиночной скорости, а буфер рекуррентных состояний растёт на 788 МиБ.
  Потолок - 42.32 t/s, и не достигается он из-за деления целевого ubatch:
  `llama_memory_hybrid::init_batch` зовёт `split_equal` с `n_keep_tail = 1 + n_rs_seq`
  ([src/llama-batch.cpp](src/llama-batch.cpp)), и слоты едут в одном ubatch только когда длины
  драфтов совпали до токена. Замер: те же два слота на одинаковом тексте дают 45.66 t/s, на
  разном - 37.98, разница 20.2%. Из неё 917 мс из 4532 - лишние пересборки графа, остальное -
  21.5 мс фиксированной стоимости на каждый лишний проход цели.
- Выравнивать длины драфтов между слотами, чтобы ubatch слился, не окупается. Проверено правкой в
  [common/speculative.cpp](common/speculative.cpp): слот, упёршийся в `p-min`, продолжал драфтить,
  пока драфтит хоть один другой, и в конце все обрезались до самой длинной «естественной» длины.
  `np2-free` стал 29447.53 и 30173.95 мс против 26273.81 и 26960.08, по стенке 31.89 против
  35.43 t/s. 361 добитая позиция купила 30 принятых токенов: предельная приёмка 8.3% при пороге
  окупаемости 48% (12.2 мс на позицию против 25.7 мс на проход цели вместе с пересборкой графа).
  Правка откачена, числа и контроли в [scripts/win-qsa/README.md](scripts/win-qsa/README.md).
- `p-min` 0.6 лучше и 0.7, и 0.5. Числа и методика замеров:
  [scripts/win-qsa/README.md](scripts/win-qsa/README.md).

## Vulkan

- Разреженный flash attention на префиле: апстримовский путь (`flash_attn_sparse_compact.comp`)
  строит точный список ячеек KV на строку маски, но включается только на декоде, потому что
  тайл из нескольких строк запроса один список разделить не может. Наше дополнение - две формы
  тайла на префиле, обе дают тот же результат, что и плотное ядро.
  `GGML_VK_FA_SPARSE_DISABLE=1` выключает разреженный путь целиком.
  Наш прежний гатер по тайлу (список на группу строк, ёмкость с запасом, откат на плотный путь
  при переполнении) снят в пользу апстримовского: он давал префилу +34% при `-ub 512` и +107%
  при `-ub 2048`, но результат отличался от плотного, а апстримовский список точный.
- Замер на Strix Halo, Qwen3.8-Flash-Next Q4_K_XL, `pp2048 @ d65536`: 223.88 против 181.82 т/с
  при `-ub 512` и 247.40 против 131.31 т/с при `-ub 2048`. На нулевой глубине разницы нет.
  Узел flash attention занимает 73.9 мс против 116.2 мс, весь граф 2.06 с против 2.57 с.
- Разреженный путь не теряет точности. `test-backend-ops` на форме модели, скалярный путь:
  NMSE против CPU 4.69e-5 при `n_kv_max=2048` и 4.75e-5 при 2052, тот же плотный путь на той же
  маске - 5.49e-5. То есть разреженное ядро ближе к точному, чем плотное, а не дальше.
  06 при этом даёт KLD 0.013119 +/- 0.000230 и совпадение top-1 95.556%: это расстояние между
  двумя ядрами, каждое из которых одинаково далеко от точного, а не потеря качества. Качество
  меряет 05: отношение PPL двух рук 1.000848 +/- 0.001475, то есть неотличимо от единицы.
- Первая форма, объединение строк тайла: пре-пасс собирает строки маски всего тайла в один
  список, тайл остаётся многострочным и матричное ядро сохраняется. Объединение точное: шейдер
  по-прежнему применяет маску для каждой строки за индирекцией списка, поэтому ячейка, которую
  данная строка не выбрала, читает `-inf` и не влияет на результат. Ёмкость списка
  `min(Br * n_kv_max, KV)` - жёсткая верхняя оценка, переполнения и отката на плотный путь нет.
  Хвост из `-1` пропускается целым блоком (`fa_sparse_tail`). Памяти столько же:
  `CEIL_DIV(nem1, Br) * Br * n_kv_max` против `nem1 * n_kv_max`. Это не тот прежний гатер,
  который менял выбор.
- Вторая форма, тайл в одну строку: тюнер пересобирает тайл в одну строку (`block_rows == 1`,
  скалярный путь), список точный и длиной ровно `n_kv_max`. Матричное ядро при этом теряется,
  поэтому нужен порог по глубине.
- Выбор между ними сборка делает сама: сначала объединение, если оно окупается, иначе тайл в
  одну строку, иначе плотное ядро. Объединение окупается при `KV >= Br * n_kv_max`, то есть с
  той глубины, где худший случай списка короче KV. Размер тайла в пороге обязателен: без
  множителя порог пропускает случаи, где ёмкость упирается в KV - обход тогда не короче
  плотного, а пре-пасс и индирекция уже оплачены. На Strix Halo (coopmat1, `Br = 16`, бюджет
  2052) это `KV >= 32832`, то есть объединение включается на глубинах от 32K.
  `GGML_VK_FA_SPARSE_GROUP` двигает порог (`0` - никогда, `N` - при `KV >= N * Br * n_kv_max`),
  `GGML_VK_FA_SPARSE_ROW_RATIO` - порог тайла в одну строку (по умолчанию 8). Плечи `g` и `r`
  в 04, 05, 06 и 07 меряют обе формы по отдельности.
  `test-backend-ops -o FLASH_ATTN_EXT`: все случаи с `n_kv_max > 0` проходят в обеих формах.
- Замер объединения на Strix Halo, 20.09. 04, `pp2048 @ d65536`: 308.53 против 255.72 т/с у тайла
  в одну строку при `-ub 512` и 296.30 против 262.83 т/с при `-ub 2048`, против плотного ядра
  в 1.53 и 2.21 раза. 07, `pp512 @ d64000`, `-ub 2048`: узел flash attention 44.5 мс против
  73.8 мс, граф 1.55 с против 1.90 с. Качество не двигается: 05 даёт одну и ту же PPL
  4.2435 +/- 0.03626 на всех трёх разреженных плечах, 06 - один и тот же KLD 0.012960 до
  шестого знака, то есть объединение не добавляет к точному списку ничего.
  Порог сначала был `min_ratio * Br * n_kv_max` (2 на coopmat1), и на 64000 токенах KV 64512
  не дотягивал 1.8% до 65664: сборка брала форму, которая на 24% медленнее. Худший случай
  `Br * n_kv_max` пессимистичен, потому что соседние строки запроса выбирают почти одни и те же
  ячейки, поэтому множитель убран. Синтетика 02 показывает предел этого: когда строки маски не
  делят выбор вообще (`sparse_grp=1`), объединение вырождается в плотный обход и проигрывает
  точному списку в 1.4 раза; при `sparse_grp=16` и выше выигрывает в 3.4 раза.
- Следующий за `MUL_MAT_ID` узел `MUL` (`ffn_moe_weighted`) применяется прямо при записи
  результата матмула, убирая запись и обратное чтение всего результата. В апстриме этот фьюз
  включён только для мат-вектора, то есть работает на декоде и не работает на префиле. Включён по
  умолчанию, `GGML_VK_MMID_SCALE_EPILOGUE_DISABLE=1` возвращает старый путь. Отказ на coopmat2:
  там эпилога в шейдере нет. Портировано из `halo-box/strix-llama.cpp` `f4d6f9e8d`.
  Замер на Strix Halo (15, шесть прогонов на каждое состояние), `pp2048`: +2.7% при `-ub 512` и
  +2.3% при `-ub 2048`, на глубине 65536 +1.1% и +1.2%. По операциям (07): `MUL` теряет 47
  вызовов и 17.3 мс, `MULTI_ADD` ещё 10.6 мс, сам матмул прибавляет 7.9 мс, граф 2.058 -> 2.042 с.
  Корректность: `test-backend-ops -o MUL_MAT_ID_FUSION` и `-o MUL_MAT_ID`, 921 случай `OK`.
  Оттуда же брали `GGML_VK_MMID_WG256` (256 потоков на тайлы `mul_mat_id`), но он сидел в ветке
  для AMD с coopmat и не проприетарным драйвером, то есть на целевом железе не выполнялся вообще;
  разница в его пользу оказалась разбросом между прогонами. Убран вместе с ветвлением.
- Загрузка весов на UMA идёт прямо в отображённый буфер. Синхронный `ggml_vk_buffer_write_2d` умеет
  писать в host-visible назначение напрямую, а асинхронный `ggml_vk_buffer_write_2d_async` проверял
  только источник, поэтому на UMA каждый тензор шёл через закреплённый staging, `copyBuffer` и
  `vkQueueSubmit`. Под WDDM submit обязан сделать резидентными все аллокации, на которые ссылается,
  и около 104 ГиБ занятой памяти это периодически роняло загрузку драфт-модели с
  `vk::Queue::submit: ErrorOutOfDeviceMemory`. Теперь `ggml_backend_vk_set_tensor_2d_async` пишет
  через `deferred_memcpy` в отображённый буфер, и на загрузке весов submit не делается вовсе
  ([ggml/src/ggml-vulkan/ggml-vulkan.cpp](ggml/src/ggml-vulkan/ggml-vulkan.cpp)).
  Быстрый путь ограничен назначениями с `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`. У планировщика через
  тот же вызов идёт копия экспертов MoE ([ggml/src/ggml-backend.cpp](ggml/src/ggml-backend.cpp)), и
  там CPU-копия перед submit переставила бы запись относительно работы, которая уже в полёте.
  Вторая половина обязательна: `ggml_backend_vk_event_record` был единственным местом "закрыть
  контекст и отправить", которое не сбрасывало `in_memcpys`. Без сброса при
  `async_use_transfer_queue == false` отложенная копия не успевает до сигнала семафора, загрузчик
  успевает перезаписать закреплённый буфер, и веса портятся молча.
  Проверено на трёх сочетаниях: Intel UHD 630 (`uma: 1`) с WEIGHTS - 8 из 8 копий по быстрому пути,
  данные верные; тот же прибор с usage ANY - 0 копий; AMD Radeon Pro 5500M (`uma: 0`) - 0 копий.
  `test-backend-ops -o SSM_SCAN` на Vulkan1 даёт 12 падающих случаев, и список посимвольно
  совпадает до и после. Закреплённые буферы на 256 МиБ остались: чтобы загрузчик писал прямо в
  отображённую память, нужно менять его самого, а `ggml_backend_vk_buffer_get_base` отдаёт
  `vk_ptr_base`, то есть `tensor->data` - смещение, а не указатель.
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
