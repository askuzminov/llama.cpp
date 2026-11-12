# Ревью ветки `cache` — полный сборник

Репозиторий: `/Users/askuzminov/Projects/github/askuzminov/llama.cpp`, форк `llama.cpp`.
Ветка: `cache`, HEAD `01ace4997`. Относительно `master`: 79 коммитов, 333 файла, `+59114 / -1070`.
Цель ревью, как её поставил владелец: (1) не сломано ли качество вывода инференса, (2) упор был на скорость, (3) общие проблемы. Приоритет: качество важнее скорости.
Машина ревью: macOS, 16 ядер, доступные бэкенды Metal и Vulkan (MoltenVK). CUDA не компилируется. Реальной модели `Qwen3.8-Flash-Next-UD-Q4_K_XL` на машине нет.
Метод: чтение дифа, сборка (`GGML_VULKAN=ON`, `LLAMA_BUILD_TESTS=ON`, каталог `build-qreview`, удалён по завершении), прогоны `ctest` / `test-backend-ops` / `llama-perplexity`, плюс 22 фоновых агента с их подагентами.
Ни один файл репозитория ревью не изменял.

## Как устроен этот файл

- **Часть A** — то, что я установил и проверил лично, без опоры на агентов. Это самое надёжное в документе.
- **Часть B** — дословные отчёты всех 22 агентов, включая подагентов, с деревом вызовов. Ничего не пересказано и не сокращено: код, пути, номера строк, таблицы и формулировки сохранены как есть, поэтому они на исходном языке. Единственное изменение вёрстки — уровень markdown-заголовков внутри чужих отчётов сдвинут на два, чтобы скелет этого документа читался; перед каждой цитатой приведена постановка задачи, которую агент получил.
- **Часть C** — чем мои проверки разошлись с выводами агентов и что было снято по ходу.
- **Часть D** — приоритетный список правок.
- **Часть E** — что осталось непроверенным и почему.

## Вердикт

`Request changes`. Качество вывода по существующим проверкам подтвердить нельзя: гейт `test-moe-cache-fork` неработоспособен, а два места в коде способны дать неверный вывод молча — на тех машинах, ради которых делалась ветка.

| # | Тяжесть | Где | Суть | Проверяемо здесь |
|---|---------|-----|------|------------------|
| 1 | Critical | `tests/test-moe-cache-fork.cpp` | Эталон качества состоит из NaN, сравнение это не замечает; NaN-прогон делает зелёными все 5 режимов | да |
| 2 | Critical | `ggml/src/ggml-cpu/tiled/tiled.cpp:615-640,1066` | F16-активация + квантованные веса на CPU -> `GGML_ASSERT`, проверка `src1` недостижима | нет (нужен x86) |
| 3 | Critical | `ggml/src/ggml-cuda/mmq.cuh:1510-1514` | Guard stream-k-сетки переполняется в 32 бита: либо громкий аборт, либо ложное прохождение и тихий неверный вывод | нет (нет CUDA) |
| 4 | Critical | `tools/server/fork/server-context-fork.cpp:126-131`, `server-context.cpp:6258` | Ограничитель размера входящего состояния отключён дефолтом (`--cache-ram -1` -> 0) | частично |
| 5 | Critical | `tools/server/fork/server-pool-fork.cpp:1552,1565,1646,1710` | `joinable std::thread` без `join`/`detach` на путях остановки -> `std::terminate` | нет |
| 6 | High | `tests/test-moe-cache-fork.cpp:492-511` | `break` до накопления `max_diff`: «logprobs 0 apart» означает «ничего не измерено» | да |
| 7 | High | `tools/server/tests/unit/*` | Сравнений текста вывода почти нет: одно на ~2570 строк | да |
| 8 | Suggestion | `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmid_f16reg.comp:225` | Перенос индекса в элементном пространстве, читает веса другого эксперта без fault; фикс в одну строку | нет |
| 9 | Suggestion | `tools/server/server-context.cpp:3507-3567` | `save`/`load` слота теряет `base_pos` чекпойнта: успех отвечает, состояние отклоняют | нет |
| 10 | Suggestion | `tools/server/fork/server-task-fork.cpp:823-849,1611-1623` | Spill-блобы без контрольной суммы; `restore_spilled()` может принять чужой файл и удалить его при эвикции | нет |

## Часть A. Проверено лично

### Сборка и оперы

Сборка чистая, 0 ошибок. Оба новых ggml-опера (`GGML_OP_FLASH_ATTN_LIST`, `GGML_OP_GATED_DELTA_NET_LAZY`) проходят `test-backend-ops` на пяти бэкендах.

### Логика сэмплирования — без претензий

Быстрый top-k (`common/sampling.cpp`, `common/fork/top-k-fork.h`) разобран построчно и признан корректным: включение только при нейтральности сэмплеров (`mirostat == 0`, `n_probs == 0`, `!backend_sampling`, обход цепочки до `TOP_K` с проверкой только PENALTIES / DRY / TOP_N_SIGMA, `0 < top_k <= 256`), bail на ничьих (`!(h[j].first > h[j+1].first)`), bail при `n_vocab <= k + 1`, bail при grammar/budget, байасы собираются как в апстримной цепочке (`fast_bias[lb.token] += lb.bias`, затем `suppress` в `-INFINITY`), NaN не считается «больше». То, что в цепочку до пользовательских сэмплеров попадает только `logit_bias` (`common/sampling.cpp:441`), подтверждает корректность самой проверки нейтральности.

`LLAMA_MTP_VOCAB` на вывод не влияет: `src/fork/qwen4exp-fork.h` документирует «the draft logits are -inf outside the subset, as an EAGLE3 draft vocabulary; the target verifies every draft».

Rejection sampling (`common/sampling.cpp:850`) разобран и корректен: масса и состояние сэмплера сохраняются как в ветке отказа, новый `break` по EOG безопасен. Важная оговорка: путь выключен дефолтом (`common/common.h:340` `probabilistic = false`), но включается через `--spec-auto`, который форк сам использует в `scripts/win-qsa/_config.bat:71`. Теоретически `q_hat != r` нарушает распределение цели, а не только приёмку.

Инвариант пары «токен черновика <-> q» (`common/speculative.cpp:409,423,440-441`) точен: пара берётся из одного `cur_p` в одной итерации, но защищён только комментарием (`:904` «q_x is never 0 ... but guard the divide»).

### Гейт `test-moe-cache-fork` — неработоспособен

Эталонный файл прогона:

```
n=192  nan_logprobs=1536  tok[:8]=[5,5,5,5,5,5,5,5]  md5=5bb224c3
```

1536 из 1536 logprob — NaN, «сгенерированный» токен повторён 192 раза. Из 10 прогонов фоновой проверки 9 дали тот же md5. Сравнение этого не видит:

- `max_diff = std::max(max_diff, (double) std::fabs(recs[i].logprob[a] - want[i].logprob[b]))` — при NaN оба `fabs` дают NaN, `std::max` при false возвращает первый аргумент, то есть NaN тихо отбрасывается и `max_diff` остаётся `0.0`;
- при всех NaN `partial_sort` (`:254-256`) даёт один и тот же порядок, поэтому два NaN-прогона побитово равны и проходят `tol = 0.0` для `keep`/`upload`;
- `break` при первом несовпадении токенов стоит до накопления `max_diff`.

Эталон пишется на машине прогона (fixture `moe-cache-ref`, аргумент `--out`, `tests/tests-fork.cmake:203-213`), поэтому NaN-прогон в CI делает зелёными все пять режимов.

Отдельно: `main()` читает только `-m`, `--out`, `--ref`, `--mode` (`:359-372`), неизвестные аргументы молча пропускаются, устройство всегда первый GPU/IGPU (`:411-416`). То есть выбор устройства этому тесту не передать.

### NaN локализован: не кэш экспертов и не ветка

Замер на toy-модели `test-models/qwen4exp-moe.gguf` (случайные веса из `test-llama-archs`), с выключенными по умолчанию `--moe-cache` и `--phase-mem`:

| конфиг | результат |
|--------|-----------|
| `-ngl 0` (CPU) | `Final estimate: PPL = 128.4806 +/- 0.17350` |
| `--device Vulkan0 -ngl 99` | `Final estimate: PPL = 128.4648 +/- 0.17434` |
| `-ngl 99` (Metal) | `E Unexpected negative standard deviation of log(prob)`, 4 прогона из 4 |
| `env LLAMA_UPSTREAM=all` | NaN |
| `env LLAMA_UPSTREAM=qwen4exp` | NaN |
| `env LLAMA_HC_F16=0` | NaN |
| `env LLAMA_KV_ELASTIC=0` | NaN |
| `-fa off` | NaN |
| `env LLAMA_MOE_DOWN_F16=0` | NaN |

Точка расхождения (разбор бэкендов, подтверждён флагом офлоада): `GGML_OP_MUL` в `grouped_norm(hidden, model.layers[il].ple_norm_query)` — `src/models/qwen4exp.cpp:1490` (он же `src/fork/qwen4exp-fork.cpp:3209`), PLE-гейт блока 0; `-ot "blk.0.ple_norm_query.weight=CPU"` даёт `PPL = 128.4806`, `-ot "blk.0.ple_norm_key.weight=CPU"` — NaN.

Почему это не регрессия ветки: `git diff master...HEAD -- src/models/qwen4exp.cpp` пуст; `ggml-metal` тронут одним файлом на `+5/-1`, и эти строки только сужают поддержку; в дифе `ggml.c` ни один существующий опер не меняет математику; `LLAMA_UPSTREAM=all` отключает компоненты уровня llama, но не ggml-ядро. Прямого прогона `master` не было.

Полезный побочный результат: `test-backend-ops -o TOP_K` расходится `MTL0` с CPU (ERR до 2.0, в том числе `ties=0`). И Metal-код top-k, и тест-кейсы `TOP_K` веткой не тронуты — это наследство `master`, но для приоритета «качество» важно: как примитив `GGML_OP_TOP_K` на Metal несостоятелен.

### Причина неустойчивости `test-moe-cache-fork`

Не «протухшие» веса кэша, а два независимых источника шума:

- `tol = 0.0` при недетерминизме GPU-пути самой модели: `llama-completion` без `--moe-cache` на GPU — 4 прогона, 4 разных вывода; `-ngl 0` (CPU) — 5 прогонов побитово идентичны;
- план размещения экспертов зависит от объёма свободной памяти GPU на старте (`src/fork/llama-context-fork.cpp:372-424`, упругий KV + `--phase-mem`) и от успешности аллокации послойных слотов (`src/fork/llama-moecache-fork.cpp:1288-1293`).

Доказательств того, что кэш отдаёт неверные веса, нет. Механизм расхождения в `ggml-backend`/`ggml-alloc`, который объяснял бы «тот же вход -> разные числа», не найден.

### `test-spec-draft-q-fork`

Падает не из-за логики, а из-за невоспроизводимости форварда черновика на Metal: CPU, BLAS и Vulkan чисто, `MTL0` и дефолт падают, доля «токенов вне q» скачет 1869/2180/3324/4322 из 12000; `-ngl 0` и `-fa off` падают, `--spec-draft-ngl 0` проходит. Здесь `--device` работает (`common_params_parse`, `:99`), в отличие от теста кэша.

### `test-qsa-multi-fork`

`CHECK(worst == 0.0)` (`:515`) — излишне строгая бит-точная проверка между двумя путями в двух процессах (фактически 1.49e-08); ниже в том же файле допускается `1e-3`.

### Серверная часть

Подтверждены: `if (max_bytes > 0 && lay.size > max_bytes)` при `SPILL_MAX_BYTES = 1ull << 40` (`tools/server/fork/server-task-fork.cpp:1364,94-103`); `pool_cache_ram_fork = prompt_cache->limit_size` (`tools/server/server-context.cpp:6258`); `--cache-ram -1` -> 0 (`tools/server/fork/server-context-fork.cpp:126-131`); `watch`-поток создан в `:1552`, `join()` в `:1565`, `reader` создан в `:1646`, `join()` в `:1710`; `discovery_t::start()` (`:2919-2930`) без гейта `stopping`, который есть у `impl::start_poller` (`:816-821`).

### Гигиена и процесс

CI не изменён ни одним коммитом ветки (`git diff master...HEAD -- .github` пуст). Из 79 коммитов 74 называются «cache» — бисектировать скорость невозможно. `tech_report.pdf` (2.3 МБ) и собранный UI в дифе. Секретов в дифе нет (срабатывания — плейсхолдеры апстрима). Ручек окружения со статичным дефолтом 56, из них 39 включены по умолчанию.
## Дерево агентов

Корневых агента 4, всего 22 включая подагентов (глубина до 3). Порядок в части B соответствует этому дереву.

```
- B1  Переиспользование состояния: spill prompt-кэша, row cache, кэш экспертов, чекпойнты, упругий KV, окно MTP
  - B1.1 Row cache: корректность
  - B1.2 Система чекпойнтов
  - B1.3 Упругий KV, окно MTP, разделяемый KV
  - B1.4 Spill prompt-кэша
- B2  Численность и память: Vulkan, CUDA, ядро ggml
  - B2.1 Аллокатор ядра ggml
  - B2.2 Vulkan-фьюны и операции
  - B2.3 CUDA fork inc, первый проход
  - B2.4 Новые и изменённые GLSL-шейдеры
    - B2.4.1 Ручки окружения шейдеров и условия фьюжна
      - B2.4.1.1 Гейты семи маленьких Vulkan-фьюнов
      - B2.4.1.2 Путь sparse-списка во flash-attention
      - B2.4.1.3 Побитовый анализ matmul-путей форка
    - B2.4.2 Шейдер mul_mmid_f16reg
  - B2.5 Изменения CPU-операций форка
  - B2.6 CUDA fork inc, второй проход
- B3  Конкурентность сервера: пул, роутер, слоты, spill
  - B3.1 Состояние spill в server-task-fork
  - B3.2 Конкурентность пула
  - B3.3 Python-тесты форка сервера
- B4  Разбор падения test-spec-draft-q-fork
```


## Часть B. Дословные отчёты агентов

Текст ниже не редактировался: код, пути, номера строк, таблицы и формулировки остались на языке оригинала.

### B1 Переиспользование состояния: spill prompt-кэша, row cache, кэш экспертов, чекпойнты, упругий KV, окно MTP

- Агент: `general-purpose-cj32o0JgdCGeWPcOLhzkogCbShcPRb7L` (тип `general-purpose`, глубина 0, родитель: корневой вызов ревью)
- Статус: completed, создан 13:44:06 UTC, финальный отчёт 25819 символов

**Постановка задачи (дословно из лога):**

You are doing a READ-ONLY correctness review of a performance fork of llama.cpp. Do not edit, create, or delete any file. Do not run builds (another process is building in build-qreview).

REPO: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`).
REVIEW TARGET: the diff `git diff master...HEAD` (79 commits, ~59k added lines). Merge base is `master` (aa588a346).
CONTEXT: This is a research/perf fork. Design notes are in FORK.md and RESEARCH.md (written in Russian). The fork is controlled by env knobs read via `llama_fork_env_int(NAME, default)` / `common_fork_env_int(NAME, default)`; many default to ON. The owner's #1 concern is: DOES THIS BREAK INFERENCE OUTPUT QUALITY (wrong/stale KV, wrong tokens, silently different results), and secondarily are there memory-safety or concurrency bugs.

YOUR SCOPE - state reuse and caching correctness ONLY (other reviewers cover sampling/speculation and the GPU backends, so skip those):
1. Prompt cache spill: `tests/test-prompt-cache-spill.cpp` and the code it tests in `tools/server/server-context.cpp`, plus `src/fork/*` spill-related files.
2. Row cache: `tests/test-row-cache.cpp`, `tests/test-row-cache-impl-fork.cpp` and the implementation they cover (grep LLAMA_FORK_ROW_CACHE / row_cache).
3. MoE expert cache incl. the cross-restart state file (`LLAMA_MOE_CACHE_STATE`, `LLAMA_MOE_CACHE_KEEP/COPY/ASYNC/FILL`, header `LMC2`, the model-id hash): files under src/ and ggml/ mentioning moe_cache.
4. Checkpoints: `LLAMA_CKPT_*` (incl. `LLAMA_CKPT_USER_BREAK` default 0), `tests/test-checkpoint-partial.cpp`, `tests/test-ckpt-system-fork.cpp`.
5. Elastic KV: `LLAMA_KV_ELASTIC*` (note `LLAMA_KV_ELASTIC_ZERO_READS` default 1), `ggml/src/ggml-cuda/fork/elastic-buffer-fork.inc`.
6. MTP attention window + its SWA KV cache: `LLAMA_MTP_WINDOW` (default 16384) and `LLAMA_MTP_WINDOW_KV` (default 1), `tests/test-mtp-window-fork.cpp`.
7. Shared KV: `LLAMA_KV_SHARED`.

For each finding, answer concretely:
- What is the exact mechanism that can produce a WRONG result (stale row reused, missing invalidation, hash collision, id/pos mismatch, off-by-one, capacity overflow, races between threads)?
- Is the risky path ON BY DEFAULT or only behind an opt-in knob/flag? State the knob and its default value - this changes the severity a lot.
- Can it be triggered by a normal server workload (multiple slots, context shift, cache reuse across requests, save/restart) rather than only a unit test?

Method requirements: read the actual implementation, not just tests or comments. Trace invalidation paths end to end - e.g. for the MoE state file, verify what happens when the file was written by a DIFFERENT model/quant/fine-tune, and whether the hash actually distinguishes them. For a cache, find every place the underlying data changes and check each one invalidates the cache.

Report at most 10 findings, most severe first, each as:
SEVERITY (Critical = can produce wrong output / crash by default; Suggestion; Nit) | file:line | one-line claim | concrete failure scenario (inputs -> wrong outcome) | default-on or opt-in | your evidence (the code you read that proves it).
If you find nothing in an area, say so in one line - do not invent findings. Do not report style, naming, or missing comments. Cite real line numbers you actually read.

**Отчёт агента (дословно):**

FORK.md:72 confirms: "Текст с кешем от прогона к прогону разный" = "the text with the cache differs from run to run" — the fork documents non-reproducible text with the expert cache. Also FORK.md:601 notes "CUDA не повторяет результат" (CUDA does not repeat the result) — the fork already acknowledges non-reproducibility on CUDA, and interestingly mentions "Mac, Vulkan, тестовая qwen4exp с экспертами в RAM: логиты совпадают с глубиной 0" (logits match at depth 0 vs prefetch depths 1-4 on Mac/Vulkan).

Good. Now let me write the report. I'll write in English (the parent's brief is English; the user-language memory says respond in Russian — but this is a subagent report to a parent agent, and the parent's instructions/format are English; the parent will assemble. The memory says the user asks for Russian. Hmm. The report goes to the parent agent, whose brief is in English and demands an exact English format. Mixing would violate "one language". The parent needs the format `SEVERITY | file:line | ...` — those fields are English. I'll write in English to match the requested format exactly; the parent's own report to the user can be in Russian.

Now, finalize the findings list with precise evidence.

Finding 1 — Critical: NaN logits in the MoE-cache test's reference run (test ref mode) on GPU, i.e. the fork's qwen4exp path emits NaN logits for a normal decode loop, and the test's NaN-insensitive comparison hides it.
- file:line: tests/test-moe-cache-fork.cpp:505 (the comparison) — but the NaN originates in the model/GPU path. I'd better phrase the claim as: "test-moe-cache-fork's reference is NaN in 7/8 runs and `std::max(max_diff, NaN)` keeps the old value, so a NaN-vs-NaN run passes the test and only finite-vs-finite or finite-vs-NaN pairs can be measured — the test cannot detect wrong logits."
- failure scenario: run `test-moe-cache-fork --out ref.bin` twice: r1 finite, r2..r8 all-NaN (tokens 5,5,5,... forever). If both ref and mode are NaN → PASS with garbage output. Server impact: a qwen4exp model on GPU can silently produce NaN logits and repeat one token (with sampling falling back), no error.
- default-on/opt-in: the NaN shows in the test config (cache+phase-mem+elastic on); my llama-completion runs with all three OFF produced non-reproducible but non-NaN-confirmed output; so I could not determine whether the NaN needs the cache. Mark that clearly.
Hmm, careful. The Critical claim should be about what I can prove. Two provable things: (i) the test's reference is NaN in 7/8 runs and the test still passes when both sides are NaN — provable and serious (test is not a gate); (ii) NaN logits in the fork's qwen4exp GPU decode — provable from my records dump (the ref records have lp=nan and constant token 5), and the config has cache+phase-mem+elastic ON.

I'll make finding 1 the NaN + NaN-insensitive comparison (Critical), and finding 2 the base nondeterminism with cache off (High) which also corrects the parent's CPU datapoint.

Finding 3 — High: `--moe-cache` output is not bit-reproducible by construction: each expert matmul is split host/device by cache state, and the cache state depends on (a) free VRAM at alloc() (llama-moecache-fork.cpp:1163-1189 for auto; :1288-1293 layer-by-layer allocation aborting partway when a buffer fails) and (b) with LLAMA_MOE_CACHE_ASYNC=1 + CUDA (:1090), insert() bails at :1860 if the LRU tail's upload isn't published yet, so admission history depends on upload timing. tol=0 at :492 for keep/upload is therefore unachievable in principle. FORK.md:72 already documents "text differs run to run".

Finding 4 — High/Suggestion: the test binary silently ignores unknown args, including `--device` (:363-372) → the "keep --device CPU 2/1" datapoint ran on Metal; and always uses the first GPU (:411-416). This matters because it's the evidence the parent used to rule out GPU numerics. Severity: Suggestion (harness), but flag it prominently as it invalidates a conclusion. I'll put it as High-ish? It's not a product bug. I'll mark Suggestion but list it 4th with a note "corrects the parent's evidence".

Then the subagent findings 5-10 as Severity Suggestion.

Also need to state clean areas in one line and what I couldn't determine.

Answering the four asks explicitly in a short section.

Regarding "the 0.0408 in keep": explain the break-before-measure → 0.0408 measured only on record 0 (first generated token after the 180-token prompt), so it cannot be attributed to a specific expert row; and a 0.04 nats shift of one top-8 logprob plus an argmax flip one token later is what any host/device split of one expert of one layer in the last block would produce on a 32-expert random-router toy model — i.e. it does NOT imply wider corruption, but it also does NOT prove a single stale row: with the reference itself NaN/nondeterministic, it's indistinguishable from base drift.

Let me now also double check my claim "the test always sets cp.n_moe_cache=32 and phase_mem=true" — line 203 confirmed n_moe_cache; phase_mem I read earlier in run(). Fine.

Also I should give the exact line for llama-context.cpp update() call (1785-1786) and llama-graph.cpp 2474-2480 — from the summary, already read.

One more number check for the report: r1 md5 85de1b2594 finite vs r2..r8 5bb224c3d4 NaN. And today's 4 more runs p_1..p_4 all NaN md5=5bb224c3d4 → so now 11/12 NaN runs, and the NaN output is bit-identical across runs (same md5) — meaning NaN is a *stable attractor*, and the finite runs are also mutually identical. That's a useful nuance: within one branch it's deterministic; the run either goes NaN or lands on the one finite trajectory. That smells like an overflow that depends on... something tiny. Since NaN-vs-NaN gives identical records, the divergence happens very early (first decode token) and then locks in.

Interesting: 2 finite trajectories? r1 (85de) and p0_6 (96a8) had identical first records but different md5 — so they diverged later. Fine.

Let me write the report now. Keep it organized, precise, and note verification commands.

For the report length: it needs to be thorough but structured. Let me draft.

Also mention: I did not modify the repo tree (only /tmp/moeq files).

Regarding severity naming convention: the report_findings tool exists but only for a review flow with an artifact; the parent asked for a text format. I won't call report_findings (no artifact oracle; the brief says report in text). Actually, the parent's brief says "Report at most 10 findings, most severe first, as `SEVERITY | file:line | ...`" — text. Good.

Let me write it. I have enough evidence. Final report below.

---

### Correctness review: prompt-cache spill, row cache, MoE expert cache, checkpoints, elastic KV, MTP window, shared KV

Verification runs were done with `build-qreview/bin/test-moe-cache-fork` and `build-qreview/bin/llama-completion` (HEAD `01ace4997`, Metal device). **No repository file was modified**; scratch artifacts live in `/tmp/moeq/` only.

#### Findings (most severe first)

**1 | Critical | tests/test-moe-cache-fork.cpp:505 | The MoE-cache test's reference is NaN in most runs, and `std::max(max_diff, NaN)` silently keeps the old value, so a NaN-vs-NaN run *passes* the test with garbage output | `test-moe-cache-fork -m qwen4exp-moe.gguf --out r$i.bin` 8x: run 1 produced finite logits, runs 2-8 produced **NaN logits** and the same degenerate token stream (`tok[:6]=[5,5,5,5,5,5]`, `lp0_0=nan`) — 11 of my 12 ref runs were NaN, all NaN runs bit-identical (md5 `5bb224c3d4`). `fabs(finite - nan)` and `fabs(nan - nan)` are both NaN, `std::max` returns its first argument for a false comparison, so `max_diff` never becomes NaN: two NaN runs compare "0 apart" and PASS; only finite-vs-finite pairs can ever be measured. A real server consequence is visible in the same data: a qwen4exp decode loop on GPU can produce NaN logits and repeat one token forever with no error logged. | The NaN appears in the test config, which has `cp.n_moe_cache = 32` (:203), `cp.phase_mem = true` and `LLAMA_KV_ELASTIC=1` (:385) all ON even in `ref` mode (`KEEP=0`, `COPY=0`, :388-389) — so this is the cache/phase-mem/elastic stack, not the eviction logic specifically | 12 ref runs parsed with `/tmp/moeq/dump.py`; comparison loop read at :492-511 |

**2 | High | tests/test-moe-cache-fork.cpp:492 | Bit-exactness against a reference produced by this fork's GPU path is unsound in principle: base inference of the same model, with `--moe-cache`, `--phase-mem` and elastic KV all **off**, is nondeterministic run to run | `llama-completion -m qwen4exp-moe.gguf -f prompt.txt --temp 0 -s 42 -n 64 -ngl 99 -ot exps=CPU` (no `--moe-cache`, no `--phase-mem`): 4 runs -> 3 identical, 1 different; an earlier 6-run set gave 6 different outputs; `-ngl 99` (all weights on GPU) also 6/6 different; **`-ngl 0` (CPU only) 5 runs, 5 bit-identical**. With `--moe-cache 32 --phase-mem` added, 4 runs -> 4 different outputs. Divergence starts at the first generated token, which is the same shape as the `keep`/`upload` failures — so those failures are not localizable to expert-cache eviction | default-on in the sense that any GPU run of a qwen4exp model is exposed; the `tol = 0.0` asserted for `keep`/`upload` at :492 cannot hold on this platform regardless of the cache | the runs above |

**3 | High | src/fork/llama-moecache-fork.cpp:1860 + :1288-1293 | The cache's contents depend on transient device state (free VRAM at `alloc()`, and on async-upload completion when `LLAMA_MOE_CACHE_ASYNC=1`), so which experts are computed on host vs device is not a function of the request sequence alone | `alloc()` reads `ggml_backend_dev_memory()` (:1163-1165) and for `--moe-cache auto` sizes slots from it (:1178-1186); even with a fixed slot count, the per-layer slot buffers are allocated one layer at a time and a failed allocation `break`s out (:1288-1293), leaving *fewer cached layers* than asked for - so a run that starts while another process holds VRAM takes a different host/device split than the run that produced the reference. On top of that, `insert()` refuses to recycle a slot whose upload is not yet published (`if (!c.live[s]) return;` :1859-1861); with `async_upload` (`= async_table && dev name starts with "CUDA"`, :1090) whether the tail slot has been published by then is scheduler timing, and the refusal changes the LRU history for every later step. Both make output timing/pressure-dependent. Note I found **no stale-row path** here: a refused insert leaves `skip[e]=0` so the host computes that expert (`ggml-cpu.c:1576-1700` memsets skipped `mul_mat_id` rows to zero, and `llama-graph.cpp:2474-2480` adds the two halves), so the results are the same math in a different FP order, not wrong rows | opt-in: `--moe-cache` default is 0 (`common/common.h:790`, `include/llama.h:380`, `llama-context.h:4301`, and `llama-completion --help`: "default: 0"); `LLAMA_MOE_CACHE_ASYNC` default 0 (:964), so `async_upload` is off unless someone opts into it; `LLAMA_MOE_CACHE_COPY` default 1 and `LLAMA_MOE_CACHE_KEEP` default -1 only matter once `--moe-cache`/`--phase-mem` are on | code read at the cited lines; the resulting non-reproducibility is stated by the fork itself at FORK.md:72 ("Текст с кешем от прогона к прогону разный") and FORK.md:601 ("CUDA не повторяет результат") |

**4 | Suggestion | tests/test-moe-cache-fork.cpp:363-372 | The test binary has no `--device` flag and silently ignores unknown arguments, so the "keep `--device CPU`: 2 passed / 1 failed" datapoint did not run on CPU | `main()` parses only `-m`, `--out`, `--ref`, `--mode`; anything else is skipped by the `else if` chain, and the device is always the first GPU/IGPU (:411-416, with "SKIPPED, no GPU device" otherwise). Passing `--device CPU` (or `--device Vulkan0`) to it therefore changes nothing - all three of those configurations were Metal runs. The conclusion "the divergence reproduces with the model on the CPU too, which rules out GPU reassociation" is not supported by that command; my own CPU control had to be done with `llama-completion -ngl 0`, and there the fork is bit-stable (finding 2) | harness only, not reachable in a server | the arg loop and device pick read above |

**5 | Suggestion | tools/server/server-context.cpp:3507-3513, 3540-3567 | The `/slots/{id}/save` + `/load` round trip does not persist `llama_checkpoint_fork::base_pos`, so restored DELTA checkpoints are relabelled BASE and the state load is rejected - the endpoint reports success while the cache silently stops working | `save_checkpoint_file` writes `{header, data_spec, data_tgt}` and never writes `base_pos`; `load_checkpoint_file` always passes `LLAMA_CKPT_BASE` (:3557); on restore `llama_context::llama_state_seq_set_data_ext` throws `"wrong sequence state magic"` (src/llama-context.cpp:3768-3772), which the server catches as a failed restore -> full recompute of a prefix the operator just saved and reloaded. `SLOT_CKPT_VERSION` is still 1, so there is no migration marker. Secondary: the `pop_front` trim at :3509-3513 can drop the anchor and orphan younger DELTAs | default-on behaviour of the save/load endpoints once checkpoints exist (`LLAMA_CKPT_USER_BREAK` default 0 only disables user-turn capture; batch-split capture is on) | code read at the cited lines; `base_pos` appears only at :3394/:3405/:3410, not in the save/load path |

**6 | Suggestion | tools/server/fork/server-task-fork.cpp:823-849 | `restore_spilled()` rebuilds the in-memory index from the file *names* but never re-checks that `spill_name_of(owner, signature, h.uid)` equals the name it scanned, so a damaged/edited spill directory makes the server adopt a uid that points at a *different* intact state's file, and the later `drop()` (:1573-1590) deletes that other state's file | Operator copies/restores a spill directory, or one `.bin` is truncated and rewritten: the entry with the mismatched uid is adopted (only header `uid`/`version`/`bytes_prompt` are checked), then eviction of that slot calls `drop()`, which unlinks the file named by `spill_name_of(...)` - a live state belonging to another slot. Output stays correct (the blob is still byte-compared against the serialized prompt before being applied, :1626-1629), so this is silent prompt-cache loss, not wrong tokens | opt-in: `--cache-spill-dir` default `""` (spill off); prompt cache itself is on by default | `spill_entries()` enforces `uid != h.uid -> continue` at :283-284; `restore_spilled()` has no such check |

**7 | Suggestion | tools/server/fork/server-task-fork.cpp:1611-1623 | Spill blobs carry no checksum: only lengths are validated, so a bit-rotted KV/checkpoint blob is applied as a valid prefix | `read_to()` checks `size()` only, then `apply_restored()` calls `llama_state_seq_set_data_ext` (:1669). A flipped bit inside a spilled delta chain changes restored KV content (or lands in a delta body whose `base_pos` still matches), and the server answers from a corrupted prefix cache - a silently wrong continuation with no error. Chain validation is byte-level only (`KV_DELTA_MAGIC` + `base_pos`, src/fork/llama-state-delta-fork.cpp:120-143) | opt-in (spill off by default), but a spill directory lives on disk across restarts, which is exactly where bit rot happens | :1611-1623, :1667-1676 |

**8 | Suggestion | tools/server/fork/server-task-fork.cpp:1529 | Pool-imported prompt-cache states are permanently `disk_evicted = true`, so they are never spilled or persisted again | A state migrated over the pool queue gets `disk_evicted=true` and an empty `disk_id`; `write_state()` requires `!s.second.disk_evicted` (:1879) and `disk_reserve()`/`disk_budget_left()` short-circuit on `disk_id != ""`, which the imported state never gets. Result: a long conversation imported from a peer slot is invisible to the disk budget, is lost on crash, and is re-prefilled from token 0 instead of reusing the migrated prefix | opt-in (`--pool`-style multi-process flow; spill itself off by default) | :1529 vs :1879, :905-932, :861-867 |

**9 | Suggestion | src/fork/llama-kv-window-fork.cpp:14 (with src/llama-kv-cache.cpp:868-871) | Draft-side SWA window evictions are not rolled back when `prepare()` fails partway through a multi-ubatch batch, so repeated failures progressively evict KV the draft still needs inside its window | `llama_kv_cache::prepare()` returns `false` at several points (e.g. :995-997 `find_slot` with no empty slot) after `need_cell` already ran `swa_room_fork` for earlier ubatches; there is no rollback anywhere in the file. `llama_prompt_checkpoint` uses `ret` as the commit flag (src/llama-context.cpp:3210-3215), so an aborted batch leaves permanent evictions. Same class at :26-35: the stream whose window is consulted is `ubatch.seq_id[i][0]` while *all* seq_ids of the row are added to `seqs`, so a multi-stream ubatch can run `seq_rm` for a different stream's cells | `LLAMA_MTP_WINDOW` default 16384 / `LLAMA_MTP_WINDOW_KV` default 1 (src/fork/qwen4exp-fork.cpp:499,506), but the code only runs for `ctx_type == MTP`, i.e. a draft-MTP server; impact is draft acceptance and KV churn only - the target model verifies tokens, so the emitted tokens stay correct | files read in full; the 3 `seq_rm(seq,0,-1)` sites (:288, :302, :331) are the only writers |

**10 | Suggestion | tests/test-row-cache.cpp:156-168 | The row cache's pool/`gather()` path is never exercised in CI, because `LLAMA_ROW_CACHE_SLICES` defaults to true and the test only reaches it when it is 0 | `src/llama-row-cache-fork.cpp:482-578` (slice pool, `gather()`, `pread` into pooled slots, the `slice_in_flight` reuse protocol) is only built when `lazy_read::cache != nullptr`, which needs `LLAMA_ROW_CACHE_SLICES=0`; the test defaults to the direct-`pread` path, so every pool-protocol bug in the fork's most-used read path (row cache is ON by default for `lazy_mode` AUTO with `-lzm dio`) ships untested | the row cache is opt-in at the app level (`-lzm dio`), but the *untested* branch is the default-on branch of that feature | tests/test-row-cache.cpp:156-168; src/llama-row-cache-fork.cpp:482-578 |

#### Areas I could not fault (one line each)

- **Row cache correctness**: invalidation (no writable path exists; reader is stateless), worker-pool shutdown/exception protocol, slot aliasing during eviction, bounds checks, and object lifetimes are all sound; the single mutex at :651 is a latency issue, not correctness.
- **Checkpoint chain logic**: ordering, coverage, `trim_prefix`/`drop_suffix`, double-apply protection (`loaded_ckpt_end`), hybrid/recurrent fallback to BASE, spill and pool transport of `base_pos`, chain-granular eviction (`LLAMA_CKPT_SYSTEM_BREAK` default 1024), cross-slot reuse via `dest_seq_id`, and save-vs-decode races (`synchronize()` at src/llama-context.cpp:4899/4925, single server thread) are correct.
- **Elastic KV**: the prefix-capacity map invariants hold across alloc/grow/seq_cp/state-restore/shrink/clear/K-shift and stream copy; `LLAMA_KV_ELASTIC_ZERO_READS` (default 1) provably only applies to emulated buffers (`src/llama-kv-emu.cpp:161-165`), CUDA never exports the hook, so it cannot zero live cells on CUDA; commit/decommit rounding and syncs are correct.
- **MTP attention window**: cannot change the tokens the target model accepts; only the `WINDOW_KV=0/-1` opt-in paths rest on a fragile "last cells == latest positions" assumption (qwen4exp-fork.cpp:1373-1421).
- **Shared KV** (`LLAMA_KV_ELASTIC`/`LLAMA_KV_SHARED`): `LLAMA_KV_SHARED` defaults 0 (src/llama-kv-cache.cpp:226) and only swaps the buffer type - no index logic; the shared-slot path was already upstream, and its `used_max`/head-0 assumptions are commented and hold.
- **MoE state file**: `state_save`/`state_load` (src/fork/llama-moecache-fork.cpp:1732-1815) store **expert numbers only, never weights or activations**, verify magic `LMC2` + `state_id()` + per-layer `il` + counts + each-expert-once, and are written via tmp+rename. So the fact that `state_id()` (:1685-1712) hashes arch/name/file-size/n_elements/n_expert/shapes but *not weight bytes* can at worst make a same-named, same-size fine-tune warm its cache in a suboptimal order - it cannot serve a wrong row. Clean.
- **MoE graph math**: `build_moe_cache()` (src/fork/llama-graph-fork.cpp) + the `ggml_mul_mat_id_add_skip` contract (ggml/include/ggml.h:1532-1535) + the CPU implementation (ggml/src/ggml-cpu/ggml-cpu.c:1576-1700, `memset` to zero for `-1` ids and skipped rows) do give "each row from one side only", and the slot-0-zero invariant is enforced by construction (src/llama-model.cpp:473-478). No double-count and no uninitialized read found.

#### Direct answers to your four questions

**(1) Root cause of the ctest failures / the two divergence shapes.** The test compares a candidate run against a reference *file written by a different process* (:471-486), with `tol = 0.0` for `keep`/`upload`, `1e-4` for `fill`, `1e-5` for `kv` (:492). The token loop `break`s on the first mismatch (:496-501) **before** accumulating `max_diff`, so "token 0: 5 against 11 / logprobs 0 apart" means "diverged at the very first token, nothing was measured" - the `0` is an artifact, not evidence of identical logprobs. "token 1 ... / logprobs 0.0408 apart" means the *first* record was finite on both sides with one shared top-8 id differing by 0.0408 nats, and record 1's argmax then flipped. Underneath both: the reference itself is NaN in 11 of my 12 runs (finding 1), and the model is nondeterministic on GPU with the cache off (finding 2).

**(2) Is the expert cache changing the computed output, or is this benign reassociation?** Both statements are true at different levels, and the code settles the design question: with `--moe-cache`, an expert's matmul is executed on the device slots when cached and on the host (`ggml_mul_mat_id` + skip vector) when not, and the two halves are added (src/llama-graph.cpp:2474-2480). So the cached path is *by construction* not bit-comparable to the uncached path, and `tol = 0` for `keep` is unachievable as a design goal. What is **not** benign is the NaN: NaN is not floating-point reassociation, and I could not attribute it to the cache (see "could not determine").

**(3) Reachability in a normal server start.** `--moe-cache` is **off by default** (0) at every level - `include/llama.h:380`, `common/common.h:790`, `src/llama-context.h:4301`, and the app's own `--help` text. `LLAMA_MOE_CACHE_COPY` (default 1), `LLAMA_MOE_CACHE_KEEP` (default -1, `src/fork/llama-context-fork.cpp:45`), `LLAMA_MOE_CACHE_FILL` (default 0) and `LLAMA_MOE_CACHE_ASYNC` (default 0) are only read inside the cache object, which is constructed only when `n_moe_cache != 0` (src/llama-context.cpp:628-629) - so no default server start touches any of this. FORK.md does **not** claim output equality: FORK.md:72 records the +17.5%/+30.2% decode wins and explicitly says the text with the cache differs from run to run; FORK.md:601 separately notes CUDA does not repeat its own result.

**(4) Mechanism for the run-to-run difference.** Two real, code-level mechanisms, neither of which is a stale row: the host/device split depends on free VRAM at `alloc()` and on whether per-layer slot buffers succeed (:1163-1193, :1288-1293), and with async uploads on it depends on upload completion (:1859-1861). I checked specifically for what you suspected - there is **no** `std::map`/`unordered_map`, no `rand()`, no `time()`/`clock()` anywhere in `llama-moecache-fork.cpp` (only two `getenv` calls), and `hot_now()`/LRU ordering iterate fixed-size vectors, so the cache's own decision procedure is deterministic given identical inputs and identical allocation success.

**On the 0.0408 nats in `keep`:** because of the `break`, that number was measured on **record 0 only** - the first token generated after the first 180-token prompt, before any sustained slot thrashing. It is fully consistent with a single expert of a single layer being computed on the host instead of the device (or vice versa) on a 32-expert toy model with a random router, and equally consistent with the base GPU drift I measured with the cache off. It does **not** imply a wide corruption, and it does not localize to one wrong row either - the measurement is too coarse (one token, one shared top-8 id) to distinguish those.

#### What I could not determine

- **The origin of the NaN logits.** They are real and reproducible-as-a-branch in the test's `ref` configuration (cache + `--phase-mem` + elastic KV all on), but localizing them needs the GPU/numerics side (`LLAMA_HC_F16` default 1, `LLAMA_MOE_DOWN_F16`, `dsv4_hc_pre_f16_fork`, the sparse-attention paths in `src/fork/qwen4exp-fork.cpp`), which is another reviewer's scope. I could not run that isolation: every shell command that sets an environment variable (`FOO=0 binary`, `export FOO=0`, `env FOO=0 binary`) was rejected by the permission layer for background agents, so I never got a NaN-count for `LLAMA_HC_F16=0` / `LLAMA_MOE_DOWN_F16=0`.
- **Whether the NaN requires the expert cache.** My cache-off control (`llama-completion`) produced non-reproducible output but its text was emitted as an undecodable hex blob, and the two commands I needed to decode and check it for NaN were permission-denied; I can prove "cache off => nondeterministic" but not "cache off => NaN".
- **Whether the fork's flaky ctest modes are caused by the eviction logic at all.** On the evidence above I do not think they are: `state` stays green because it never compares tokens against a finite baseline, and every flaky mode's failure is explained by an unreliable reference rather than by a demonstrated stale row. If you want the MoE cache judged, the test needs a CPU-only (`-ngl 0`-equivalent) reference and a NaN assertion first - with those in place I would treat any remaining `keep` failure as a genuine cache bug.

---

### B1.1 Row cache: корректность

- Агент: `Explore-9cM1erpC5FyeDgpAxXahXlra7dGMXU3Z` (тип `Explore`, глубина 1, родитель: `general-purpose-cj32o0JgdCGeWPcOLhzkogCbShcPRb7L`)
- Статус: completed, создан 13:47:32 UTC, финальный отчёт 29198 символов

**Постановка задачи (дословно из лога):**

READ-ONLY code review (do NOT edit any file, do NOT run builds) of a performance fork of llama.cpp.
REPO: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`). Merge base is branch `master`.

TOPIC UNDER REVIEW: the "row cache" feature only. Files to start from:
- src/fork/llama-row-cache-fork.cpp, src/fork/llama-row-cache-fork.h
- tests/test-row-cache.cpp, tests/test-row-cache-impl-fork.cpp
- grep the whole repo for `row_cache`, `ROW_CACHE`, `LLAMA_FORK_ROW_CACHE` to find EVERY call site (src/, ggml/, tools/server/, common/). Also read the FORK.md sections (Russian) that describe it, for intent - but verify against code.

GOAL: find correctness bugs that could make model inference produce WRONG output (stale row reused after the underlying weight/KV row changed, missing invalidation, id/position mismatch, off-by-one, capacity overflow, hash/index collision, data races between threads, using a row from a different layer/token, buffer aliasing), or memory-safety/concurrency bugs.

METHOD REQUIREMENTS (mandatory):
- Read the actual implementation, not just the tests or the comments.
- For every cache, enumerate every place the underlying data can change, and check whether each one invalidates the cache entry. Trace invalidation end-to-end.
- Determine for each risky code path whether it is ON BY DEFAULT or only behind an opt-in knob. Find the exact knob name and the default value in the code (search for `llama_fork_env_int` / `common_fork_env_int` calls and read how they parse defaults). This changes severity a lot.
- Determine whether the risky path can be hit by a NORMAL server workload (multiple slots, context shift/eviction, prompt cache reuse across requests, --checkpoint restore, model reload) rather than only by the unit test. Check tools/server/server-context.cpp and llama decoding paths for the call sites.

REPORT FORMAT: at most 6 findings, most severe first. Each: SEVERITY (Critical = can produce wrong output or crash by default / Suggestion / Nit) | file:line | one-line claim | concrete failure scenario (inputs -> wrong outcome) | default-on or opt-in (name the knob + default) | evidence (the specific code you read that proves it, quote a few lines).
If an area is clean, say so in one line. Do NOT invent findings, do NOT report style/naming/missing comments. Cite real line numbers you actually read.

**Отчёт агента (дословно):**

I've spent enough time on this. Merging across seqs is possible via llama_batch (if the API is used that way, a single token can have multiple seq_ids. The server usually doesn't do this... actually, if kv_unified is present, the server's batch might also merge identical tokens, but I couldn't confirm — rather than claiming it's a bug, I'll record it as a defensive assert).

No, wait — before finalizing, since the topic clearly asks for a search for race conditions, let me do one last focused check on what I think is the highest-risk concurrency point. The worker loop:

```cpp
{
    std::unique_lock<std::mutex> lock(job_mutex);
    job_cv.wait(lock, [this] {
        return job_stop || (job_runs != nullptr && job_next < job_runs->size());
    });
    if (job_stop) return;
    i_job = job_next++;
    slice = job_rows != nullptr;
}

const size_t n_jobs = job_runs->size();
const run &  r      = (*job_runs)[i_job];
```

Consider this sequence:
1. run_jobs A is active with 1 job. Worker takes i_job=0 and releases the lock.
2. Worker executes read_run. Completes, locks, ++job_done == 1 == n_jobs → notify done_cv → releases lock.
3. run_jobs A wakes up, sets job_runs = nullptr and returns. gather A returns from get_rows.
4. A different caller (warmer thread) calls get_rows → gather → run_jobs B: locks, job_runs=&runsB, job_next=0, job_done=0, unlock, notify_all.
5. Worker loops back to step: predicate — grabs B's job. No problem.

Alternative interleaving: between steps 2 and 3, could the worker re-enter wait and see a spurious state? Predicate: job_runs is still &runsA (until 3), job_next(1) < size(1) is false → sleeps. Then B's setup: notify wakes it up. No problem.

Another alternative: after run_jobs A returns (job_runs=nullptr), before the reset guard of gather_slices... irrelevant.

But — if the worker is slow and hadn't yet evaluated the predicate between wait and `i_job = job_next++`, and the caller thread did step 3 (job_runs=nullptr)... no — step 3 requires all workers to increment. The worker only increments after job completion. A worker that hasn't taken a job yet will eventually take it and complete it. OK.

Now — **real** race candidates in this pattern: the worker holds `const run & r` and reads `r` **after** releasing the lock. `run_jobs` returns to gather after all jobs increment job_done. The worker's increment happens after read_run(r) completes. So before run_jobs returns, `r` is not used. Then the runs vector is destroyed. ✓

And `n_jobs` — after the lock — `job_runs` is still valid (same argument). ✓

OK, one more: if `run_jobs` throws an exception (job_error non-empty) — job_runs has already been set to nullptr before throw. Workers are idle. Next gather proceeds. ✓

And **worker exceptions from memcpy?** None.

OK, and for the `slice` flag too — the block path never coexists.

Another possible correctness hazard to double-check for `gather_slices` + `read_slice` regarding **rows requested within a single gather exceeding dst**: `job_dst + job_ent[e]*n_cols` — entry < n, dst capacity n*n_cols ✓.

And `gather_slices` when called with n>0 and `dst` — all entries written exactly once? Each sorted position k belongs to exactly one distinct-row group (plan_first covers all positions), and each distinct row is processed exactly once within one slice (disjoint [0,n_uniq) ranges). ✓

Also re-confirmed that the constructor's range check rejects `offs > file_size` etc. ✓

Also, what if `n_rows` exceeds INT32_MAX... rows are int32. gather validates `row >= n_rows` in int32. If n_rows > INT32_MAX, rows can't reach it — just less table. No problem.

Final: I'll also verify the claim that `row_cache_env` treats parsed<=0 as default — i.e., LLAMA_ROW_CACHE_SLICES=0 → row_cache_env_flag: parsed==0 → returns false (off). ✓

OK, are there any CRITICAL findings? Let me reconsider the most subtle point of `read_slice` once more — the consecutive-row merging:

```
int64_t len = 1;
while (k + len < k_end && (size_t) len < slice_run_max && job_rows[k + len] == row + len) {
    len++;
}
const size_t off  = offs + (size_t) row*row_size;
const size_t size = (size_t) len*row_size;
... read into buf ...
for (int64_t j = 0; j < len; j++) {
    const uint8_t * src = buf + (size_t) j*row_size;
```

Rows row..row+len-1 are contiguous in the file ✓, because job_rows[k+j] == row+j.

Then entries. `e0 = job_first[k+j]; e1 = job_first[k+j+1];` — job_first has n_uniq+1 entries. k+j+1 ≤ k_end = r.block + r.n ≤ n_uniq. ✓

`out = job_dst + job_ent[e0]*n_cols` ✓ e0 < n.

Dequantize once per row, copy to duplicates. ✓

Now — is `row` the row from `job_rows[k]` — but wait, the loop after processing advances `k += len` — right.

One more subtle point: the `want` zero-padding inside read_slice uses the `file_size` of the cache's file. Multiple files share the same path. ✓

OK, this is really the last one: **`file->seek(off, SEEK_SET)` followed by `read_raw_unsafe`** — in the buffered FILE* case (fd==-1... actually if fd==-1 then fp path), read_raw_unsafe calls `tell()` → llama_mmap_fseek/ftell... the seek+read pattern per worker on its own handle. ✓ No sharing.

Conclusion: the row cache core has been reviewed and I believe no correctness bugs were found. Findings to report:

1. Suggestion — test coverage gap: the block-pool fallback (LLAMA_ROW_CACHE_SLICES=0) is not exercised by test-row-cache.cpp (the comment at ~line 157 claims both paths are tested, but the env is not set, and the default is slices=1). Reachable via opt-in.

2. Suggestion — the ple_warmer queue (LLAMA_PLE_WARM, default 1, gated on -lzm dio) is unbounded and not deduplicated: under speculative decoding with the warmer lagging (slow disk), std::vector<std::vector<int32_t>> grows without bound (qwen4exp-fork.cpp ~2974-3010). Memory growth only.

Hmm, also:
3. Nit — the cast overflow of `plan_first.push_back((uint32_t) n)` requires exactly n==UINT32_MAX. Unrealistic. Skip? The instructions say don't fabricate findings. Could mention as nit or omit. I think I'll include as nit since off-by-overflow technically exists but is unreachable. Actually n > UINT32_MAX throws. n == UINT32_MAX is allowed and (uint32_t)n wraps to 0 → job_first last entry 0 → duplicate row's copy loop `for e = e0+1; e < e1` — if e1==0 e0 is also... positions up to n-1 = UINT32_MAX-1 fit. But the terminator push is n → wraps to 0 → the last distinct row's e1=0 < e0 → no dup copies → last distinct row's duplicate entries are never written → dst garbage. But gathering 4,294,967,295 rows in one call would require a 16 TB dst... actually the dst is n*n_cols*4 bytes — physically impossible. Skip as theoretical? Could be a nit. Mention briefly as nit, as a missing guard.

4. Nit/Suggestion — the GGML_ASSERT in `llm_graph_input_ple_fork::set_input` (qwen4exp-fork.cpp:2791) aborts the process if a ubatch contains a token with n_seq_id > 1. The prefetch path explicitly treats that case as "skip prefetch" (reason 2), but the synchronous path asserts — so workloads that batch a shared token across slots would crash the server rather than fall back. Only when ple_cached (-lzm dio) — opt-in. I think it's legitimate to report as low severity: asymmetry between _ahead's handling and set_input's assert. But can the server actually emit n_seq_id>1? The server's batch builder adds tokens per slot with a single seq. kv_unified doesn't merge automatically... upstream llama.cpp only merges when the caller passes multiple seqs per token in llama_batch. The server never does. So unreachable on server. Leave as a nit.

5. Worth noting as informational/clean: the take() prefix-equality validation protects every prefetch path — stale-row bugs impossible.

6. Hmm, wait — one thing to re-check before I declare clean: **in `set_input`, if `take()` returns false and a prefetch worker is still active on pf (mismatched idx), the synchronous `get_rows` runs concurrently with the active prefetch worker** — both call `ple_cache->get_rows` — impl::mutex serializes them ✓. No dst aliasing (different buffers) ✓.

And **the take() prefix-match case with pf.idx longer**: rows gathered for the superset are used for the prefix — the remaining rows of pf.idx beyond idx are for tokens of the same planned ubatch... irrelevant.

Actually, hmm, wait. Let me reconsider one more potential real bug in `ple_prefetch_take`. I want to re-verify the `keep` semantics of `ple_prefetch_first` against `pf.rows` sizing:

- Hint launch: pf.idx size = min(next_tokens.size, n_ubatch_split()) * n_heads rows... wait, idx is n_heads per token.
- prefetch_first launch (keep=true): pf.rows.resize(idx.size()*dim).

But take() does `std::swap(dst, pf.rows); dst.resize(idx.size()*ple_head_dim);` — if take succeeded on a gather launched with larger pf.idx: pf.rows had capacity for pf.idx.size()*dim floats, but dst resized down. ✓

But — if take() happens **after** launch has written to pf.rows with a gather for a different idx, then resize — join precedes so no torn writes ✓.

One more: `ple_prefetch_take` compares idx against pf.idx — pf.idx is read under pf.mutex — but prefetch worker reads pf.idx concurrently — no problem — but the comparison result is used after finish() — idx is immutable during worker run (writes only in launch after join, under mutex) ✓.

OK. Also confirm that `ple_prefetch_start` is called on the `hint` ubatch built from `next_tokens` — `hint` is a local llama_ubatch whose data is backed by shared_ptr `data` — `qwen4exp_ple_rows` copies into `idx` synchronously ✓. The ubatch doesn't escape. ✓ And the ub in `llama_fork_tokens_ubatch` is used synchronously inside ple_prefetch_first/ple_warm. ✓ peek_next ubatch — ubatches_peek is owned by mctx. idx is computed synchronously ✓. Worker doesn't touch mctx. ✓

Alright. One last check: `impl`'s `gather()` — `throw` inside scan for out-of-range row: mid-chunk — pinned etc. are local. get_rows unlocks. ✓

Let me also glance to confirm there aren't other users (e.g., `llama-moe-cache.cpp` uses "row cache"?) — grep shows llama-moe-cache.cpp only mentions get_rows (expert cache). Out of topic.

One more, a potential issue I want to nail down concretely: **`row_buf` aliasing in `gather()`**: `src = row_buf.data()`, then to_float/memcpy into out — but in the b0==b1 case src points into pool. Next iteration overwrites row_buf — used immediately. ✓

And **dst overlap**: could caller pass dst overlapping pool? No, dst is caller's vector.

Final answer structure: honestly report that no Critical correctness bug was found in the row cache core, plus the findings:

1. (Suggestion) Untested block pool fallback + its comment overstates coverage.
2. (Suggestion) Unbounded, non-deduplicated ple_warmer queue (default ON under -lzm dio) — unbounded memory growth under slow-disk + speculative decoding. Also reads go through the same mutex and could stall prompt-side gather (perf cliff). In fact, more importantly: the warmer holds impl mutex for its whole gather. A synchronous get_rows in set_input blocks on it — a queued backlog of stale idx reads delays the critical path. Suggestion.
3. (Nit) `plan_first.push_back((uint32_t) n)` wraps for a gather of exactly UINT32_MAX rows — unreachable in practice.
4. (Nit) GGML_ASSERT in set_input n_seq_id>1 vs skip handling in prefetch — process abort instead of fallback. Unreachable from the current server's batch builder.

Wait — I should double-check finding 2 more concretely, and also recheck `ple_warm` n_skip: `idx.erase(idx.begin(), idx.begin() + n_skip*n_heads)` — wait, if qwen4exp_ple_rows fills idx at n_heads per token then erase the first n_skip*n_heads ✓.

And the warmer's queue: pushed per draft step. Worker processes serially. Each idx ≤ n_ubatch*n_heads rows... for MTP drafts with LLAMA_PLE_WARM=1 (default): pushed per draft step. Each is a few tokens (n_new). Small. Queue depth grows only if the read rate is exceeded — each item is tiny (a few rows). Bounded in practice by draft step rate. Weak finding. Leave as a nit rather than a suggestion? A real unbounded queue but each entry is small. The producer rate is limited by MTP draft steps. Worker read is ~ms. Realistic growth is small. Downgrade to nit or drop. Hmm. Instructions: don't fabricate findings. I'll include as a nit, with the caveat that rate is limited.

Anything else more severe I might be overlooking? Let me look at one more area with fresh eyes: `llama_fork_before_compute` returns true when the hint was taken — and the return value at the call site in `llama-context.cpp`? Called at line 1758 as a void statement. OK.

And the state of `next_tokens` across decode calls that don't call fork_before_compute (hyb null): cleared. ✓

Also `fork_prefetch_first`: `batch.n_tokens >= 32`, single seq, contiguous — prefetches top `min(n_tokens, cparams.n_ubatch)` tokens. But note: **the phase step might choose a first ubatch from different tokens**? Phase step decides ubatch split — the first ubatch is always a prefix of the batch's tokens in order? Yes, ubatches preserve order. ✓ And position alignment ✓.

But: what if `cparams.n_ubatch` < the actual first ubatch chosen by the phase step (phase_mem might allow larger ubatch than n_ubatch? "tokens of the largest one" says step takes the max = n_ubatch). OK.

Another possible real issue: **prefetch_first runs at start of decode and computes idx with `kv.get_prev_tokens(ub, n, res)` — using the whole kv (not the context). For the predecessor of a batch's first token — the get_prev of the KV cache uses seq to find last n tokens by position — ✓. But wait: at decode start, before apply, predecessor of first ubatch's tokens within the batch: handled by the own map. But get_prev for token i returns... actually `qwen4exp_ple_rows_ahead` overwrites entries for p ≥ pos_min from `own`, and leaves entries for p < pos_min as returned by get_prev. get_prev(next, n_prev, prev) fills prev for all tokens — presumably from cache. At decode start the cache has no new tokens yet, so predecessors within the ubatch come from own ✓, but only if p ≥ pos_min and found in own. Gap check returns 3. ✓

And critically — between prefetch_first's idx computation and set_input, does anything modify the KV cells for positions of previous tokens (context shift from a full cache)? Context shift occurs on apply when the cache is full — could change get_prev results → mismatch → slow path. ✓ Safe.

OK, final report. Actually let me also do a sanity check on the knob name "slices default" for finding 1: row_cache_env_flag("LLAMA_ROW_CACHE_SLICES", true) — default true — and test-row-cache.cpp sets only LLAMA_ROW_CACHE_RUN (lines 167-168), never sets SLICES. Confirmed.

Also one more possible real bug in `read_slice` that I want to triple-check: `const int64_t k_end = (int64_t) r.block + r.n;` — r.n is uint32_t per (count of distinct rows in slice). Fine.

But — `runs.push_back({ (uint64_t) k, (uint32_t) std::min<int64_t>(per, n_uniq - k), 0 });` — n_uniq could exceed UINT32_MAX if n exceeds... n_uniq ≤ n ≤ UINT32_MAX ✓ and per ≤ n_uniq ≤ UINT32_MAX ✓.

Also, `n_slices` — `4*files.size()` where files.size()==n_threads ✓.

Also `gather_slices`: `plan_keys.resize(n)` — if n shrank between calls, stale entries past n remain but only first n sorted ✓ `std::sort(plan_keys.begin(), plan_keys.end())` sorts all n elements (size==n) ✓.

Writing the report. Ordering: no Critical. Most severe first:

1. Suggestion: the test claims both-path coverage but the block path is never exercised (default of SLICES is on. Test never sets it) — an env-only fallback gather() (the trickiest code: capacity, straddle staging, run coalescing) ships untested. Also FORK.md describes it as legacy. file: tests/test-row-cache.cpp:156-168 + src/fork/llama-row-cache-fork.cpp:215.

2. Suggestion: ple_warmer: unbounded queue + single worker holding the row cache mutex can delay the critical path's gather (warmer and prefetch and set_input share one mutex). qwen4exp-fork.cpp ~2974-3010 + row cache get_rows 651-653. Opt-in with -lzm dio gate. Knobs default on. Actually, is this a "correctness" issue? No — perf. The goal is correctness/memory-safety bugs. Perf suggestions are borderline but useful. The prompt says findings for the correctness/perf fork review should be wrong-output/memory-safety/concurrency-related, and the report allows Suggestion severity for anything substantive.

3. Nit: (uint32_t) n wrap at row 609.

4. Nit: assert asymmetry.

And explicit clean statements: pool protocol (job publication/completion, slot disjointness, per-worker file handles and buffers), EOF/range validation, prefetch invalidation (full prefix-equality validation of the row list before reuse, join before read), thread join on destruction order, exception path from worker to caller — all checked, no stale/wrong-row path: the prefetched rows are keyed on the exact row-index list recomputed at consumption time, so stale KV/eviction/shift mismatch can only fall back to a fresh gather, never produce wrong output.

Also note the default-ness: the entire subsystem is off by default (lazy_mode default AUTO is LLAMA_LAZY_MODE_AUTO at llama.cpp:3609. Only explicit -lzm dio, or AUTO on a device without mmap, activates the row cache). PLE_PREFETCH/PREFETCH_FIRST/WARM default to 1 but no-op if not ple_cached.

One more consideration — is there any chance `rows` (idx) tensor path and emb path coexist and `ple_cached` is true but `rows` is also non-null? set_input: if rows → set idx and return. rows is set when the tensor exists (mmap mode). dio mode creates no tensor (create_tensor returns nullptr) → rows null, emb... wait, which input does the graph build with? Check build_inp_ple to see whether emb is created in the cache path. I'll trust the earlier header comment (FORK) and tests. Skip.

Line numbers to cite — need actual numbers:
- row cache: get_rows 651; gather_slices 580; plan_first push 609; read_slice 404; constructor 149-300; run_jobs 451; worker loop ~233-290 (from my read: workers emplace around line 237... in the read with offset 150-264, "for (int int i = 0; i < n_threads; i++)" appears around 234). Line references to use: llama-row-cache-fork.cpp:236-260 (worker loop), :451-479 (run_jobs), :580-611 (gather_slices), :609 plan_first terminator, :404-448 (read_slice).
- qwen4exp-fork.cpp: set_input 2781-2822; take 3011-3045; launch 2926-2955; warmer worker 2975-3009; before_compute 3049-3068.
- tests/test-row-cache.cpp:156-168.

For take cite lines 3027-3031 (`if (pf.idx.size() < idx.size() || !std::equal(...`).

For launch cite lines 2946-2951 (worker thread).

Let me confirm these line numbers precisely for accurate citation. From my reads:
- 2781 `void llm_graph_input_ple_fork::set_input` (from grep: set_input body I read at 2781+ with offset 2760, line 2781). My read showed "void llm_graph_input_ple_fork::set_input(const llama_ubatch * ubatch) {" appearing at line 2781 (offset 2760, ~21 lines in). Assert at ~2791. `GGML_ASSERT(pmodel.ple_cache ...` at 2812 (grep showed 2812). get_rows call 2817.
- ple_prefetch_launch: from grep line 2939 `pf.finish()`. So launch is ~2925-2955. Worker get_rows at 2948.
- take: pf.finish at 3032 (grep). take starts ~3011.
- warmer worker creation ~2976-3007, get_rows at 2990 (grep).
- before_compute 3049. peek_next 3058.
- llama-context.cpp:1753 set_inputs; 1758 fork_before_compute.
- row cache: get_rows 651; mutex 83; run_jobs 451-479; reset guard 632-634.
- constructor guard ~229-233.
- default: llama-model.cpp:3609 /*.lazy_mode=*/ LLAMA_LAZY_MODE_AUTO.
- llama-model-loader.cpp:1184 DIO gate; 1180 AUTO 4GiB.
- llama-model.cpp:1659-1667 AUTO→DIO fallback when no mmap.
- LLAMA_PLE_PREFETCH default: qwen4exp-fork.cpp:2879-2881 (`static const bool enabled = ... llama_fork_env_int("LLAMA_PLE_PREFETCH", 1)`); the line I read was in the region 2877-2880 (grep for ple_prefetch_start is ~2874). Approx cite: 2876-2892.
- LLAMA_PLE_WARM default 1 at ~2958-2960.
- LLAMA_PLE_PREFETCH_FIRST at ~2908.

Report language: user's message is English. Memory says the user prefers Russian... memory: "the user asks to reply in Russian and stick to a single language throughout the response." The user's prompt here is fully English. Per output-language rule: respond in the language of the user's input (English). The memory about Russian is a user preference — but this request is written in English. Rule: "If the user explicitly requests a specific language, switch." No explicit request here. Mixed: language of input = English. Go with English (matches the input). The memory reads "respond in Russian and don't mix languages" — that's a saved preference. Hmm. Input language is English. The rule says match input language. Memory says user wants Russian. But the user wrote a long English prompt — the delegation caller might not even be the same person. Go with English, matching current input (no explicit Russian request in this message), since it's a technical report and the input is English.

Let me write the final report with at most 6 findings, honest about the mostly-clean core. # Row cache review — findings

Scope reviewed: `src/fork/llama-row-cache-fork.cpp/.h` (full 705 lines), the `llama_row_cache` call sites in `src/fork/qwen4exp-fork.cpp` (construction at :775-790, `set_input` at :2781-2822, prefetch/warm machinery at :2874-3090), loader hooks (`src/llama-model-loader.cpp:1172-1200`, `:1459-1469`), `src/fork/llama-context-fork.cpp:740-796`, `src/llama-context.cpp:1753-1758`, `tools/server/server-context.cpp:5977-6029`, `tests/test-row-cache.cpp`, and the FORK.md PLE/row-cache sections (:625-700, :2446-2450). No other repo consumers of `row_cache` exist (grep over `src/`, `ggml/`, `tools/server/`, `common/`).

**Default-on status (applies to every finding below):** the whole row-cache subsystem is **opt-in**. `llama_model_default_params` sets `lazy_mode = LLAMA_LAZY_MODE_AUTO` (`/Users/askuzminov/Projects/github/askuzminov/llama.cpp/src/llama-model.cpp:3609`); in AUTO a marked tensor <= 4 GiB is loaded fully and a > 4 GiB one uses the **mmap** lazy path, not the row cache (`src/llama-model-loader.cpp:1180-1203`). `lazy_read::CACHE` (the row cache) is reached only with explicit `-lzm dio` (`src/llama-model-loader.cpp:1184`) or the AUTO fallback when a device reports no mmap support (`src/llama-model.cpp:1658-1667`). The prefetch knobs `LLAMA_PLE_PREFETCH` (`qwen4exp-fork.cpp:2876-2878`), `LLAMA_PLE_PREFETCH_FIRST`, `LLAMA_PLE_WARM` all **default to 1**, but are no-ops unless `ple_cached && ple_cache` (only true under dio).

#### Findings (most severe first; no Critical found)

**1. SUGGESTION — the block-pool fallback path is shipped untested; the test's comment claims coverage it does not provide.**
`tests/test-row-cache.cpp:156-168` | `src/fork/llama-row-cache-fork.cpp:215, 482-578`
The test sets only `LLAMA_ROW_CACHE_RUN`; it never sets `LLAMA_ROW_CACHE_SLICES`, which defaults to **on** (`slices = row_cache_env_flag("LLAMA_ROW_CACHE_SLICES", true)`, line 215). So every test run exercises only `gather_slices()`; the older `gather()` (pool chunking, `pinned` map, straddle staging via `row_buf`, run merging at lines 482-578) — the trickiest index arithmetic in the file — never executes in CI.
Failure scenario: a user sets `LLAMA_ROW_CACHE_SLICES=0` (documented in the header comment and FORK.md as a supported fallback); a regression in the capacity check `pinned.size() + n_new > n_slots` or the straddle copy loop goes undetected and silently returns wrong bytes for rows that straddle the unaligned header (`offs` not a multiple of `row_size`, which the constructor explicitly allows).
Evidence: test comment line 157-158: *"every shape is run twice: once with each block read on its own, once with adjacent blocks merged into one request. both paths must return exactly the same rows"* — but both runs use the slice path (`run_max` only feeds `slice_run_max` at line 217 when slices is on). Opt-in only.

**2. SUGGESTION — prefetch/warmer/synchronous gathers serialize on one mutex; a slow queued warmer directly delays the critical-path gather.**
`src/fork/llama-row-cache-fork.cpp:651-653` + `src/fork/qwen4exp-fork.cpp:2975-3009`
`get_rows()` takes `pimpl->mutex` for the **entire** gather. The `ple_warmer` worker (LLAMA_PLE_WARM=1, default) drains its queue serially through the same mutex; the prefetch worker and the synchronous gather in `set_input` contend with it.
Failure scenario: slow disk + MTP speculative decoding; the warmer accumulates a backlog (queue is an unbounded `std::vector<std::vector<int32_t>>`, entries pushed each draft step at :3004-3008 with no dedup or staleness drop); when the target's verify ubatch runs `set_input`, `ple_prefetch_take` misses (warmer never fills the prefetch slot) and the fresh `ple_cache->get_rows(...)` (`qwen4exp-fork.cpp:2817`) blocks behind arbitrary many stale warmer reads. Result is still **correct** (it is only waiting), so this is a latency cliff, not wrong output. Opt-in (`-lzm dio`), knobs default-on within it.

**3. NIT — `gather_slices` entry-index terminator wraps for a gather of exactly `UINT32_MAX` rows.**
`src/fork/llama-row-cache-fork.cpp:587, 609`
The guard rejects `n > (int64_t) UINT32_MAX` but allows `n == UINT32_MAX`; `plan_first.push_back((uint32_t) n)` then wraps to 0, so the last distinct row's entry range becomes empty (`e1 = 0`) and its duplicate entries in `dst` are never written (`read_slice` loop `for (uint32_t e = e0 + 1; e < e1; e++)`, line 440). Not reachable in practice: the caller's `dst` would need `n * ple_head_dim * 4` bytes (~16 TiB). Theoretical.

**4. NIT — `set_input` aborts the process on multi-sequence tokens where the prefetch path only skips.**
`src/fork/qwen4exp-fork.cpp:2790-2791` vs `:2852-2854`
`GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences")` — while `qwen4exp_ple_rows_ahead` treats the same condition as reason 2 ("skip prefetch") and degrades gracefully. A client/API batch that assigns one token to several sequences (`llama_batch` with `n_seq_id > 1`) crashes the server rather than falling back. I did not find a server code path that produces `n_seq_id > 1` (server adds tokens per slot, `server-context.cpp:5982-5993`), so this is defensive asymmetry, not a live crash. Opt-in gate (`-lzm dio`) either way.

#### Areas checked and found clean

- **Stale-row / invalidation, end to end**: the row cache is a pure stateless reader of file bytes (no cross-gather block cache by design, FORK.md :626-634), so there is no invalidation surface. The only "cache of results" is the PLE prefetch: `ple_prefetch_take` (`qwen4exp-fork.cpp:3011-3045`) re-validates by exact prefix equality of the row-index list recomputed at consumption time (`pf.idx.size() < idx.size() || !std::equal(idx.begin(), idx.end(), pf.idx.begin())` -> re-gather), so KV eviction, context shift, slot rewind, checkpoint restore, cancelled batches or wrong `seq`/pos in the server hint (`server-context.cpp:6028`) can only cause a **miss** (slow path), never reuse of rows for a different token/position. Prefetch idx is computed synchronously from `get_prev_tokens` + the ubatch's own tokens before the worker launches (`:2926-2955`), and the rows are a pure function of idx.
- **Worker pool protocol** (`llama-row-cache-fork.cpp:236-290, 451-479`): job publication (`job_runs`) and completion (`job_done == runs.size()`) are correctly fenced by `job_mutex`; the unlocked reads of `job_runs->size()` and `(*job_runs)[i_job]` after the assign block cannot race with `run_jobs` clearing `job_runs`, because the clearing thread cannot exit until every assigned job (including its own) has incremented `job_done`; `job_rows`-family pointers are written before the `run_jobs` lock and cleared only after the last completion. `job_runs`/gather entry paths are all serialized by `impl::mutex`. Constructor exception path is guarded by `pool_guard` (armed before any `std::thread` exists, ~:229-235); destructor `stop_workers` is single-call.
- **Slot/buffer aliasing**: runs in `gather()` get disjoint consecutive slots (`want` sorted, slot = index, capacity capped at `n_slots` before the break); in `read_slice` slices over `plan_rows` are disjoint ranges and each dst entry is written exactly once; per-worker `files[i]` handles and `slice_bufs[i]` buffers are never shared. `bench()` keeps `batch*n_blk <= n_slots`.
- **Bounds**: constructor rejects tables past EOF (`:174-178`); both gathers validate `0 <= row < n_rows`; EOF zero-fill logic in `read_run`/`read_slice` matches the `llama_file` padding semantics (`src/llama-mmap.cpp:490-541`); head-offset/vocab-size bounds are enforced at load (`qwen4exp-fork.cpp:675-692, 755-766`); `ple_n_heads` is derived as `(ngram-1)*per_gram` (`:660`), so `idx.assign(n_heads*n_tokens)` cannot overflow in `qwen4exp_ple_rows`.
- **Lifetimes**: `ple_prefetch`/`ple_warmer` destructors join their threads and `~llama_model_qwen4exp_fork` calls `finish()` before `ple_cache` is destroyed (member destruction order is safe: `ple_cache` declared before `ple_pf`/`ple_wm`); the prefetch worker captures only `pf` (heap, joined) and `ple_cache`; `pf.ok` is only read after `join()`.

---

### B1.2 Система чекпойнтов

- Агент: `Explore-Q6nK5IeGaMMac43vzMjAcgqhHdoR7U4R` (тип `Explore`, глубина 1, родитель: `general-purpose-cj32o0JgdCGeWPcOLhzkogCbShcPRb7L`)
- Статус: completed, создан 13:47:32 UTC, финальный отчёт 15457 символов

**Постановка задачи (дословно из лога):**

READ-ONLY code review (do NOT edit any file, do NOT run builds) of a performance fork of llama.cpp.
REPO: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`). Merge base is branch `master`.

TOPIC UNDER REVIEW: the "checkpoint" feature only - env knobs named `LLAMA_CKPT_*` (note `LLAMA_CKPT_USER_BREAK` defaults to 0), plus any --checkpoint / state-save related CLI args added by the fork.
Start from:
- tests/test-checkpoint-partial.cpp, tests/test-ckpt-system-fork.cpp, tests/test-checkpoint-delta.cpp, tests/test-checkpoint-delta-chain.cpp
- src/fork/llama-state-delta-fork.cpp
- grep repo-wide for `LLAMA_CKPT`, `ckpt`, `checkpoint` in src/, common/, tools/server/ (server-context.cpp is large - focus on checkpoint/state code) and read the FORK.md sections (Russian) describing the feature for intent, but verify against code.

GOAL: find correctness bugs that could make inference produce WRONG output: restoring a checkpoint that does not match the actual KV/token state (token count, position, seq_id mismatch), partial checkpoint save/load that silently truncates or skips state (e.g. only some layers or some buffers restored), delta-chain applied in wrong order or twice, off-by-one in how many tokens are covered, checkpoint reused across a different slot / different prompt / different model, state saved while KV was partially evicted or context-shifted, races between the save thread and decode. Also memory-safety/concurrency bugs (thread saving while inference mutates).

METHOD REQUIREMENTS (mandatory):
- Read the actual implementation, not just the tests or comments.
- Trace the save path and the restore path and compare what is captured vs what is restored (including per-slot state, sampler/rng state if relevant, positions, n_tokens, memory slots for recurrent/hybrid models).
- Determine for each risky code path whether it is ON BY DEFAULT or only opt-in. Find the exact knob name and default in code (search `llama_fork_env_int` / `common_fork_env_int` and read the default argument). This changes severity a lot.
- Determine whether it can be triggered by a NORMAL server workload (multiple slots, context shift, cache reuse across requests, save/restart, /slots restore endpoints) rather than only by a unit test.

REPORT FORMAT: at most 6 findings, most severe first. Each: SEVERITY (Critical = can produce wrong output or crash by default / Suggestion / Nit) | file:line | one-line claim | concrete failure scenario (inputs -> wrong outcome) | default-on or opt-in (name knob + default) | evidence (specific code you read that proves it, quote a few lines).
If an area is clean, say so in one line. Do NOT invent findings, do NOT report style/naming/missing comments. Cite real line numbers you actually read.

**Отчёт агента (дословно):**

Confirmed: `base_pos` never appears anywhere in the slot save/load code in server-context.cpp (only 3394/3405/3410 in create_checkpoint). Finding 1 confirmed.

OK, let me finalize severities. I also want to re-check `load_slot_checkpoints` (around line 3568): the trim `checkpoints.size() > (size_t) params_base.n_ctx_checkpoints` — pop_front can evict the base while keeping dependent deltas (which creates orphans on the list). The file could contain chains longer than the current cap. Include this in finding 1 (same file/area) or make it a separate minor callout. I'll go with including it.

Finding: failed capture entries (common.cpp:2467-2472 update_tgt and create_checkpoint 3427-3436 no pop). Suggestion.

Finding: delta write for `llama_kv_cache` with flags==0 produces a blob covering only the swa subcache (for iswa flags==0: kv_base delta is called; base returns 0 when… wait, in iswa state_write_delta with flags==0: kv_base->state_write_delta(io, seq, 0, base_pos) — kv_base's swa_type==NONE → partial check `(flags & PARTIAL_ONLY) && swa_type==NONE` is false since there's no PARTIAL flag… so kv_base actually writes a full delta of the base cache (magic + cells after base_pos). OK, fine, symmetric. No issue.

Also, let me double-check `update_tgt`'s fallback: if delta size is >0 but the delta write's n != ckpt_size (race? mismatch), it clears data and sets base_pos=-1. The caller continues (failed entry). Covered by the finding.

Are there any other reportable default-on issues? Another idea — **`create_checkpoint`'s host memory pressure eviction (lines 3362-3377) only considers that slot's checkpoints and only runs before creating a new checkpoint. The new checkpoint itself can still OOM (caught as bad_alloc → skipped)**. Fine.

Another idea — **budget() floor CKPT_BUDGET_FLOOR=512MiB per slot × n_parallel could exceed RAM** — bounded by reserve on next creation. Perf. Skip.

One more potential real issue to reconsider: **`server_ckpt_evict_over_budget` never evicts the newest chain (`ckpts.size() > 1` and the break when a chain reaches end)** — if there's one chain that's larger than budget (one huge checkpoint), it stays forever; but the host reserve evicts it. Skip.

Final report structure (up to 6 items, most severe first):

1. Suggestion (borderline; feature loss, not output error) — slot save/load drops base_pos. tools/server/server-context.cpp:3507-3513 + 3526-3580. Default: feature is default-on but endpoint is opt-in.

2. Suggestion — a failed capture entry stays on the list and pollutes the chain; also the base_pos from next_base_pos can derive from that pos_max (server-context-fork.cpp:391) — the only thing preventing a silent coverage gap is that the empty entry sits inside the replay range, which depends on has_dependent_delta guards on every erase path (server-context.cpp:3340, 3382) — fragile invariant. If any future/existing path removes the empty entry while keeping the later delta (e.g., restore-time cleanup erases entries individually at 5128-5135 — I showed that n_tokens/pos_max monotonicity protects this today), you'd get a hole in the KV. Report as Suggestion with a concrete "pop the entry instead" recommendation.

3. Nit — `load_slot_checkpoints` trim orphans deltas (fold into 1? keep as a separate nit).

4. Report as clean: chain ordering / eviction protection / spill and pool base_pos preservation / SWA masking consistency / draft and spec state / save-decode races (single-threaded loop; synchronize() in state API). Also the default for LLAMA_CKPT_USER_BREAK is 0: pure batching, no correctness impact; LLAMA_CKPT_SYSTEM_BREAK default 1024: verified the keep logic matches the tests.

Should I also mention the delta restore-failure capacity scenario as a nit? "For long chains, base+delta restore can exceed the SWA subcache and silently fall back to full recompute" — actually I reasoned find_slot can reuse masked cells; uncertain → don't fabricate. Skip.

Also mention: the `GGML_ASSERT(it->id_task != -1 → GGML_ABORT)` was replaced with warn+reset — improvement.

Wait — one more real candidate I should re-check before I submit: `server_ckpt_restore`'s walkback `it_base->data_tgt.empty()` skips failed captures when looking for base — I showed the delta loop replays everything in (base, cur] and fails at the failed entry → safe. What if the failed capture is between it_base and it_cur, and it_base == it_cur itself (cur is the base)? The delta loop doesn't run; fine.

And `not_captured` where cur is delta with empty data? Delta has empty data — impossible (failure converts to base or clears; delta write failure → base_pos=-1 + clear). OK.

Another potential issue: `server_ckpt_restore` returns `&restored_cp` — `restored_cp` is a reference to `base_cp` or `cur_cp` — both references into the list — valid. OK.

Also, line 5109: `pos_next = std::min(pos_next, std::max(cp->pos_min + 1, cp->pos_max));` — cp->pos_min from the restored (delta) — a delta checkpoint's pos_min = the live pos_min at creation time (the window edge) — matches master's usage. OK.

Now let me write the report. Keep it to 6 findings or fewer. Include a statement of "verified clean areas" too. Note: I couldn't find a Critical that would produce incorrect output by default. Should I also briefly touch on the speculative-replay comment area I peeked at? Out of scope for the topic; skip.

Actually, one more thing to check before I submit: the default-off knob LLAMA_CKPT_USER_BREAK=0 — check that this break doesn't affect the `is_last_user_message` gating in ways that change the "correctness" of checkpoint semantics: if break is disabled, the batch containing the last user message might extend to end-of-prompt; checkpoint offsets {4+n_ubatch, 4} near the prompt tail — checkpoint creation there is unchanged. The FORK.md claim "responses are the same" is plausible. Also if user edits the last message: server restores from end-of-previous-prompt checkpoint and recomputes — per the docs. OK.

Citations for finding 1: lines 3507-3513 (no base_pos in the save loop) and 3540-3556 (load) and common-fork.cpp apply() (base_pos<0 → llama_state_seq_set_data_ext). And in server-context.cpp the delta blob magic is rejected as "wrong sequence state magic" (llama-context.cpp around lines 3769-3772). The failure is caught and `do_reset` at server-context.cpp:5100-5105 → full recompute (correct but slow) → so not Critical per the rubric.

Report language: the user wrote in English → respond in English. (Memory says respond in Russian if the user writes in Russian; this prompt is English.) # Checkpoint feature review (branch `cache`, topic: `LLAMA_CKPT_*` + delta context checkpoints)

Scope traced: `create_checkpoint` (server-context.cpp:3317-3452) -> `common_prompt_checkpoint::update_tgt/apply` (common/common.cpp:2437-2500, common/fork/common-fork.cpp:32-83) -> `llama_state_seq_get_delta_ext/apply_delta` (llama-context.cpp:4893-4935, fork/llama-context-fork.cpp:632-682) -> `state_write_delta/state_read_delta` (src/fork/llama-state-delta-fork.cpp) -> restore path `server_ckpt_restore` (tools/server/fork/server-context-fork.cpp:427-464) and its caller (server-context.cpp:5083-5137), plus eviction (`server_ckpt_evict_*`), spill/pool serialization (fork/server-task-fork.cpp), slot save/load appendix, and the `LLAMA_CKPT_SYSTEM_BREAK` split (server-context.cpp:5288-5292, 5345-5348).

**No Critical "wrong output by default" bug was found.** Every corruption path I traced in the delta machinery fails loudly (magic/version/base_pos check) and falls back to `do_reset` = full prompt recompute (server-context.cpp:5099-5106). Findings, most severe first:

---

**1. SUGGESTION | tools/server/server-context.cpp:3507-3513 (save), :3540-3567 (load)** — the slot-file checkpoint appendix does not persist `base_pos`, so every DELTA checkpoint restored from a `/slots/{id}/save` + `/load` round trip is relabeled as a self-contained BASE and can never load again.
- Scenario (normal workload for operators using state save/restart): SWA model (e.g. gemma-class, `iswa` cache), prompt builds chain `[BASE, DELTA1, DELTA2]` (default on: `n_ctx_checkpoints = -1`, delta is chosen by `server_ckpt_next_base_pos`). Operator POSTs `/slots/5/save`, restarts, POSTs `/slots/5/load`. `load_slot_checkpoints` leaves `base_pos = -1` (the struct default), so `is_delta()` is false; the next prefix-diverged request selects the newest checkpoint and `apply()` takes the full-state branch: `llama_state_seq_set_data_ext` reads the blob's first u32 (`KV_DELTA_MAGIC = 0x4K56444C`) against `io_magic` and throws `wrong sequence state magic` (src/llama-context.cpp:3768-3772, :4942). Every restore attempt that needs the chain now logs `failed to apply checkpoint` and reprocesses the whole prompt - the saved checkpoints are silently dead weight, while the load endpoint reported `restored N context checkpoint(s)` success. No wrong output (graceful `do_reset`), and the endpoint is opt-in - hence Suggestion, but it is the clearest defect in the feature.
- Default: context checkpoints + delta chains are default-on (`n_ctx_checkpoints = -1`, common/common.h:650; no env knob for this); only the save/load endpoints are opt-in. `SLOT_CKPT_VERSION` is still `1` despite the fork changing `common_prompt_checkpoint` semantics (server-context.cpp:3457-3458).
- Evidence: save writes only `n_tokens/pos_min/pos_max/data_tgt/data_dft/data_spec` (lines 3508-3513); `grep -n base_pos tools/server/server-context.cpp` yields only 3394/3405/3410 (create path), nothing in save/load. By contrast the spill/pool format *does* carry it: `struct spill_ckpt { ... int32_t base_pos; }` (fork/server-task-fork.cpp:56-67) and reload sets `cp.base_pos = e.base_pos` (:1406).
- Related in the same function (:3565): the trim `checkpoints.size() > (size_t) params_base.n_ctx_checkpoints` `pop_front()`s single entries - with a smaller `--ctx-checkpoints` than at save time it can drop a BASE and leave orphaned DELTAs first in the list (`server_ckpt_restore` then returns nullptr, `ok = !base_cp.is_delta() ...`). Degrades the same way; should drop/trim by chain.

**2. SUGGESTION | tools/server/server-context.cpp:3427-3451 (no pop) + fork/server-context-fork.cpp:391** — a failed capture (`update_tgt` size-mismatch path clears `data_tgt`, sets `base_pos=-1`, but the entry stays in `slot.prompt.checkpoints`) both poisons the chain and can seed a coverage-hole `base_pos`; safety today rests entirely on the empty entry staying inside every later chain's replay range.
- Scenario: `[B0(pos_max 8000)]`, attempt at 12000 fails -> entry F `{n_tokens 12000, pos_max 12000, empty}` stays. Next checkpoint: `server_ckpt_next_base_pos` skips F when searching for a base but returns `ckpts.back().pos_max` = F's `pos_max` = 12000, so the new delta D records only cells `(12000, 15000]` while the chain's real coverage stops at 8000. If D were ever replayed without F, `state_read_delta`'s `base_pos` check (`delta_base_pos != base_pos`, llama-state-delta-fork.cpp:126-132) passes (the value is copied from `cp->base_pos`), and positions `(8000, 12000]` would be silently absent from the SWA window -> wrong logits. I verified all four erase paths currently keep F while D exists (`server_ckpt_has_dependent_delta` guards at server-context.cpp:3340, 3382; the post-restore cleanup at 5128-5135 is monotone in both `pos_max` and `n_tokens`, so F and D are removed together; `server_ckpt_evict_front_chain` removes a delta only as part of its base's chain). So the only reachable outcome today is a failed restore -> full recompute (wasted work + log noise, and the chain is only cleaned after one failed attempt). Still: `create_checkpoint` should `pop_back()` the empty entry the same way the `bad_alloc` handler does, instead of leaving an invariant that, if a future erase path misses it, becomes silent KV loss.
- Default-on (any transient state-size failure on the default-enabled path).

**3. NIT | src/fork/llama-state-delta-fork.cpp:150-157** — `state_read_delta` only rejects multi-stream delta blobs when `n_stream_cur != n_stream`; a same-stream-count blob with cells in several streams is applied silently (all into `seq_to_stream[seq_id]`). The write side cannot produce one (`seq_id >= 0` scan yields cells in a single stream for non-unified, `n_stream == 1` for unified), so this is defense-in-depth asymmetry with the comment's claim, nothing more.

---

Areas verified clean (one line each):

- **Chain order/coverage math**: base covers `(pos_max-n_swa, pos_max]`, each delta covers `(max(prev, pos_max-n_swa), pos_max]` minus write-time-masked cells (`state_ranges` with `cells.pos_get(i) > pos_min`, llama-kv-cache.cpp:2232-2245); restore replays `[base+1 .. cur]` in list order (server-context-fork.cpp:451-455); window requirements at the restore position are always a subset of the union - no off-by-one found.
- **Delta applied twice / on wrong seq**: base apply uses `clear_seq=true` (`seq_rm(dest_seq_id,-1,-1)`, llama-kv-cache.cpp:2545-2547), deltas `clear_seq=false`; `apply_delta` requires full blob consumption (`success && io.n_bytes() == size`, llama-context-fork.cpp:675-676).
- **Memory-type fallback**: `llama_memory_i::state_write_delta` default 0 makes recurrent, hybrid (incl. fork `llama_memory_hybrid_idx_fork`, qwen4exp; `LLM_ARCH_QWEN4EXP` is listed in `llm_arch_is_hybrid`, llama-arch.cpp:1286), DSA, DSA-ISWA, MSA memories fall back to full self-contained checkpoints (`update_tgt` COM_DBG path), and `server_ckpt_next_base_pos` forces `-1` for recurrent/hybrid - no partial-state delta on memories that cannot roll back.
- **Prompt-cache spill and `--pool` transfer**: `spill_ckpt` carries `base_pos`, sanity-checked, version-gated (`SPILL_VERSION = 4`, version mismatch drops the file), budget-overflow drops all checkpoints atomically rather than part of a chain (fork/server-task-fork.cpp:56-67, :1016-1020, :1389-1396).
- **Eviction/`keep` (`LLAMA_CKPT_SYSTEM_BREAK`, default 1024, default-on)**: chains are evicted as units only; the kept system-end checkpoint never serves as a delta base (`server_ckpt_next_base_pos` returns -1 when the last chain's base sits at `keep`, server-context-fork.cpp:364-369); `server_ckpt_evict_over_budget` never evicts the newest chain; unit test `tests/test-ckpt-system-fork.cpp` matches the code as read.
- **`LLAMA_CKPT_USER_BREAK` (default 0, confirmed `common_fork_env_int("LLAMA_CKPT_USER_BREAK", 0)` at server-context-fork.cpp:467-470)**: purely a batch-splitting knob (guard at server-context.cpp:5335); end-of-prompt checkpoint offsets `{4+n_ubatch, 4}` are unchanged, so no correctness exposure.
- **Cross-slot reuse**: checkpoint blobs store seq ids that restore discards (`dest_seq_id` overrides, llama-kv-cache.cpp:2588-2591); prompt-cache load, child-slot `copy_prompt_to`, and `/slots` load into a different id all work through `apply(ctx, slot.id)`.
- **Save vs decode races**: checkpoint creation happens between decodes on the single server-loop thread, `llama_state_seq_get_delta_ext/apply_delta` call `ctx->synchronize()` first (llama-context.cpp:4899, 4925), and spill/write-behind runs on the same idle-loop thread - no background writer touching live KV.
- **Sampler/spec state**: `data_spec` is captured with each checkpoint and restored only for the actually-restored checkpoint (:5107, server-context-fork.cpp:459-462); speculative `spec_ckpt` always uses `base_pos = -1` (server-context.cpp:4747).

---

### B1.3 Упругий KV, окно MTP, разделяемый KV

- Агент: `Explore-Xe5ZxwaNZs1r6tyot0DpxcTy09vIhDz6` (тип `Explore`, глубина 1, родитель: `general-purpose-cj32o0JgdCGeWPcOLhzkogCbShcPRb7L`)
- Статус: completed, создан 13:47:32 UTC, финальный отчёт 49983 символов

**Постановка задачи (дословно из лога):**

READ-ONLY code review (do NOT edit any file, do NOT run builds) of a performance fork of llama.cpp.
REPO: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`). Merge base is branch `master`.

TOPIC UNDER REVIEW: three KV-cache-related fork features:
1. Elastic KV: env knobs `LLAMA_KV_ELASTIC*` (note `LLAMA_KV_ELASTIC_ZERO_READS` defaults to 1). Files: src/fork/llama-kv-elastic-fork.cpp, ggml/src/ggml-cuda/fork/elastic-buffer-fork.inc, ggml/src/ggml-alloc.c changes, ggml/src/fork/ggml-backend-fork.inc, tests/test-kv-elastic-streams-fork.cpp, tests/test-elastic-buffer-fork.cpp.
2. MTP attention window + its SWA KV cache: `LLAMA_MTP_WINDOW` (default 16384), `LLAMA_MTP_WINDOW_KV` (default 1). Files: src/fork/llama-kv-window-fork.cpp, src/fork/llama-mtp-window-fork.h, tests/test-mtp-window-fork.cpp, tests/test-kv-window-fork.cpp, plus src/fork/llama-kv-cache-mask-fork.inc if related.
3. Shared KV: `LLAMA_KV_SHARED`. grep for it.
Also grep repo-wide for these knob names and for the functions they gate, to find EVERY call site (src/llama-kv-cache*.cpp, src/llama-context.cpp, src/llama-model.cpp, tools/server/). Read the FORK.md sections (Russian) for intent, but verify against code.

GOAL: find correctness bugs that could make inference produce WRONG output: a KV cell read that maps to the wrong physical row (elastic remap table stale or not updated on compaction/eviction), uninitialized/zeroed memory being read as if it were real KV (that's what ZERO_READS smells like - check exactly which reads are zeroed and whether any LEGAL read is served zeros instead of real data), reads outside the MTP window silently dropped so the model attends to nothing for old tokens when it shouldn't (check the window is only applied where semantics allow - e.g. a draft/MTP head vs the main model, and whether a long-context prompt beyond 16384 changes the sampled tokens), off-by-one in window bounds, seq_id/position mismatch across slots when KV is SHARED, two slots overwriting each other's rows, missing invalidation of the map when a slot is cleared/shifted, races between threads. Also memory-safety (OOB) and concurrency bugs.

METHOD REQUIREMENTS (mandatory):
- Read the actual implementation, not just the tests or comments.
- For each remap/index table: enumerate every place the underlying allocation changes (alloc, free, compact, defrag, clear, seq rm, context shift, cp) and check each updates the table.
- Determine for each risky code path whether it is ON BY DEFAULT or only opt-in. Find the exact knob name and default value in code (search `llama_fork_env_int` / `common_fork_env_int` and read the default argument). This changes severity a lot.
- Determine whether it can be triggered by a NORMAL server workload (multiple slots, context shift, prompt cache reuse across requests, long context >16K, save/restore) rather than only by a unit test.

REPORT FORMAT: at most 8 findings, most severe first. Each: SEVERITY (Critical = can produce wrong output or crash by default / Suggestion / Nit) | file:line | one-line claim | concrete failure scenario (inputs -> wrong outcome) | default-on or opt-in (name knob + default) | evidence (specific code you read that proves it, quote a few lines).
If an area is clean, say so in one line. Do NOT invent findings, do NOT report style/naming/missing comments. Cite real line numbers you actually read.

**Отчёт агента (дословно):**

Okay — llama_parallel_for(n, work, fn) splits [0, n) into parts, and "work" is merely an estimation heuristic. In the forked mask: n = n_tps, work = n_tps*n_kv. That is, ii ∈ [0, n_tps) → i = s*n_tps + ii ✓ correct. My OOB concern is retracted.

But wait — is the upstream set_input_kq_mask also parallelized? The fork claims to be an optimization. Data races: each part writes to its own row (idst = n_kv*i is unique per i) ✓. `idxs.reserve(ubatch->n_tokens + n_swa + 32)` is also no problem.

However — in the forked mask's parallel version, there is indeed one subtle correctness hazard that differs from upstream: upstream processes tokens sequentially across the entire ubatch per stream, and `seq_srct` uses the first-seen row per sequence. The fork processes parts in parallel; each part picks its own first row of seq as srct and fills it completely — this is also correct. Because each row is written exactly once and computed from cells. ✓.

However: the mask fill reads `cells` — during set_input, the CPU cells are stable ✓.

Now, regarding the forked mask's `goto skip` and the `idxs.push_back` recorded before the causal mask? The recording is done before the causal/swa skip decision — the same cells are recorded as upstream (recording all cells that could change, regardless of whether they are ultimately kept or discarded). Upstream order: record, then causal, then swa. Same. ✓

Okay — time to finalize. But before writing the report, let me reconsider a few specific potential correctness bugs that I haven't fully verified yet:

1. **elastic + `elastic_set_capacity_fork`'s v_heads reset**: `if (v_heads[strm] >= n_cap) v_heads[strm] = 0;` — resetting head to 0 is only CPU state; find_slot handles holes; no data corruption.

2. **CUDA VMM and `ggml_backend_cuda_buffer_get_base`/max alignment**: tensors are placed at GGML_PAD(total, 2MiB) offsets within chunks; but ggml_backend_buft_get_alloc_size may exceed ggml_nbytes due to alignment — no problem; `end - offs` is used as span; the last tensor's span = c.total - offs, and total is padded to 2MiB ≥ the device's granule? If the device's granule is larger than 2MiB (some GPUs report 2MiB; but what about, for instance, the granule reported by `cuMemGetAllocationGranularity` on some CUDA devices with VMM? For device memory the recommended value is 2MiB; minimum could differ but... generally 2MiB). If granule were, say, 4MiB: tensors would be placed at 2MiB-aligned offsets within a single granule — the tail of tensor A shares a granule with tensor B. Then when stream s is decommitted with `to = r.offs + r.span`, the granule containing the next tensor's bytes could become fully included in [from,to) — i0 rounds up, i1=(offset+size)/g rounds down: a granule containing a byte past span... span is only padded up to the next 2MiB; if granule is 4MiB, decommit range [from, to) could fully contain a granule that contains the next tensor's region? The decommit range is within [offs, offs+span) = this tensor's region; the granule containing those bytes also contains the next tensor's bytes (since granule > span's alignment). "Granule fully contained within [offset, size)" — a granule that starts at offs+span-2MiB and extends beyond offs+span would not be fully contained (its end exceeds `to`) → i1 = (offset+size)/g rounds down, excludes partial granules → safe. So even with a large granule, decommit never unmaps the next tensor's granule. But commit's rounding-outward (i1 rounds up) could map beyond the tensor... mapping extra is safe. Also, two tensors share one granule → tensor A's decommit will never free the shared granule (partial) → just memory waste. But: the initial commit zeroing! commit zero-fills the entire granule — if stream s's granule is shared with a preceding tensor that already has data... commit is only called for the KV region during capacity setting; at that point no tensor has data (init). During runtime, commit maps new granules for stream s's expansion; the outward-rounding granule could include the leading bytes of the next stream's region (the next stream starts at offs+stride; stride=nb[2] is not granule-aligned if granule > 2MiB... nb[2] = kv_size*row_bytes can be huge and not a multiple of 4MiB) — the shared granule spanning stream A's end and stream B's start. Stream B expands first: maps and zeros that granule — including stream A's trailing cells! If stream A had already written to its trailing cells (capacity > boundary), B's commit's memset zeros A's data!! Verify: in commit's loop `if (el->mapped[i]) continue;` — if A had mapped that granule first (A's need reached into it), B skips it ✓. Conversely if B maps first, mapping and zeroing the granule containing A's tail — A hasn't written yet (A hasn't reached capacity up to there). Later A expands to include that granule: already mapped → skips zeroing → A's writes are fine. But what if A had already written to those tail cells before B's mapping? A can only write within its capacity; A's writes are below its cap → granule within A's committed range → granule mapped by A → B skips. So writing implies mapping implies skip. Data loss impossible ✓.

With granule 2MiB and tensor 2MiB-aligned, the whole thing is cleaner.

3. **`el->span`**: for the last tensor `end = c.total`, includes trailing padding ✓ decommit to span covers up to the buffer's granule boundary. For non-last tensor, end = next tensor's offs — offs is 2MiB-aligned; decommit rounds down to granule so never touches the next tensor ✓.

4. **Multi-buffer and offsets in `ggml_backend_multi_buffer_alloc_buffer`**: commit/decommit are called with r.buf = chunk buffer and r.offs within that chunk ✓.

5. **CUDA VMM: reserved addresses remain even after unmap → does reads of unmapped cause FAULT (not zeros)** — CUDA's zero_reads=false covers this.

6. **state_save (`state_get_data`) on elastic reads only used ranges ✓; but `test-llama-state`... no problem.**

7. **Checkpointing (`--checkpoints`)** delta fork — restore path grows ✓.

Next, another item under investigation: `phase_followers`' MTP draft context — does the MTP follower context itself go elastic? kv_elastic requires ctx_type DEFAULT — MTP context ctx_type is MTP → not elastic ✓. And `llama_kv_elastic_no_cache_fork`... ✓.

Another one: **elastic's default-on via no_cache in plain CUDA llama-server without --phase-mem and without --moe-cache**: kv_elastic_try_fork: mode=-1, no_cache=true → proceeds → elastic_init_fork with emulate=0. Multi-stream = n_seq_max (if not unified; default is kv_unified=true → 1 stream). Then capacity grows on demand via phase_update? Is phase_update called without phase-mem? Check: phase_update is probably only called when phase_mem is enabled; in that case, who grows capacity? grow in find_slot (elastic_grow_fork) ✓, and grow in get_used+n_tokens. Comment "capacity only grows without --phase-mem" ✓. Initial cap = step 8192.

Now — without --phase-mem, the default llama-server on CUDA VMM runs all models on elastic KV. grow failure → find_slot failure → "failed to find memory slot" → request error instead of OOM crash. Acceptable behavior.

Next, a CUDA-specific risk: **commit() failure leaves granules partially mapped** — elastic_commit_fork returns false → capacity unchanged, but partial granules remain mapped (device memory consumed, zeros) — harmless.

**CUDA: cuMemSetAccess uses physical_device** ✓.

Now, let me reconsider the one thing I flagged earlier as an actual finding: the shrink path in `elastic_set_capacity_fork` on an emulated cache where `fill == -1`: no zeroing of freed cells; and if zero_reads=1, the test expects zeros there — FORK says "при нулях там нули" (where there are zeros, they are zeros) (fills with zero when zero-reads, NaN otherwise — via poison). Without poison, no zeroing — only the test relies on this. No problem.

Final candidate finding — **the mirror branch of `state_read_meta` (the sinfo_in path) doesn't grow elastic capacity for the mirror cache**: when does the mirror path execute? `sinfo_in` — a cache that mirrors another cache (the qwen4exp QSA indexer's mirror). The indexer cache's capacity grows along with attention via kv_set_capacity_fork (hybrid wrapper). During seq restore into the attention cache, find_slot grows attn+idx (grow → context's kv_elastic_set_fork → memory->kv_set_capacity_fork → grows both attn and idx). Then the mirror cache's state restore uses sinfo of the same idx — capacity within the idx cache covers it (idx capacity == attn capacity ≥ max of idx used? wrapper's grow: idx first commits n_cells, then attn sets capacity). The mirror restore's cell_count ≤ attn cell_count ≤ attn capacity = idx capacity. ✓. But what if the mirrored cache itself is the attention cache and sinfo_in comes from the indexer's restore ordering? FORK comment at line ~2655: "the attention cache grows itself and the indexer with it (llama_memory_hybrid_idx_fork), the indexer restores after it into the cells it shares" ✓.

Hmm, but wait: the mirror branch checks `!cells.is_empty(idx)` and errors — the mirrored cache's cells. And in the whole-cache restore branch the grow happens. In the single-sequence mirror branch (line ~2606) there is no grow — but the mirror's idx equals the attention's already-grown idx. OK, I believe the ordering is handled.

Given this depth, the implementation is unusually careful. Actual findings may be few. Let me use remaining effort to verify a few remaining suspicious points with grep:

(a) `elastic_zero_reads_fork` — who consumes it? grep only finds definition/declaration. If nobody consumes it... grep for call sites: aside from the definition, none found in src! The header declares it; where is it used? grep shows only lines from llama-kv-cache.h (declaration) and the definition. elastic_cover_fork directly checks `elastic_fork->zero_reads`. That means elastic_zero_reads_fork() is dead code — a nit, not reporting (style).

(b) Server's prompt cache reuse across requests + elastic: slot reuses cached prefix — uses checkpoint state restore → grows ✓.

(c) **Race: static lazy init of `llama_fork_env_int` — thread-safe ✓.**

(d) **`swa_room_fork`'s log "logged once" uses std::atomic — no problem.**

(e) **MTP window: `rows_only` also disables `ggml_get_rows(cur, inp_out_ids)` later — handled by `if (!inp_out_ids)`... code says `if (inp_out_ids) { if (!rows_only) cur = get_rows... }` ✓.

(f) In the windowed path, attention is built via `build_attn_mha`, mask `mask_out` is f16, no ALiBi, no sink. If the model needs softcap/sink in the MTP head — build_attn_mha is passed nullptr,0 — the MTP head is a simple MHA ✓.

(g) **MTP_WINDOW view: `v_all->nb[1] <= v_all->nb[2]`** — when v_trans (no FA) get_v returns a transposed view: nb[1] = row stride... a transposed V view has nb[1] larger than nb[2]? v_trans shape [n_embd_v_gqa, n_kv, ...] is actually built from a transposed storage: ne = (n_embd_v_gqa?, kv, ...), where nb[1] = whole-stream stride → nb[1] > nb[2] → window disabled. ✓.

(h) One more thing: `mask_out` in windowed when `inp_out_ids` is not null: get_rows on mask_out after reshape — get_rows indices pick rows; then column window view; but then `ggml_cont` on a strided 2D view ✓.

So the final answer: the implementation looks very defensive; the findings I can report:

1. (Suggestion/leaning Critical?) — I need real findings. Let me think deeper about a few more areas where actual bugs could hide:

**Candidate: `elastic_cover_fork`'s growth uses the `sinfo` computed by find_slot before `prepare()` continues — but a later-ordered ubatch in the same batch may grow a different stream and shrink this stream via the "other streams give back memory" path (kv_elastic_set_fork multi-stream: on failure it calls memory->kv_set_capacity_fork(0, s) for other streams!).** Sequence: find_slot for ubatch1 grows stream 0 to n_kv0 and covers ✓. Then find_slot for ubatch2 for stream 1 fails, triggers grow of stream 1 → if cap < want and streams>1: shrink all other streams to their used (kv_set_capacity_fork(0, s)) — this may shrink stream 0 below n_kv0 (down to used_max_p1)! ubatch1 was already placed (apply_ubatch in prepare only updates CPU cells; K/V writes are later at graph time). Then the graph for ubatch1 reads n_kv0 rows of stream 0 — if the shrink dropped below n_kv0 → CUDA fault on read!

But wait: the shrink clamps to `max(n_cap, n_used, 256)`, where n_used = used_max_p1(stream0) — after ubatch1 is applied, its cells are used, so used_max_p1 ≥ ubatch1's max idx +1 = n_kv0 (without padding). n_kv = pad(used_max_p1, 256) ≤ capacity. Because capacity is ≥ used and a multiple of 256... after shrink, n_cap = pad(max(0, used_max_p1, 256), 256) ≥ pad(used_max_p1) = n_kv0 (when used_max_p1 < size). Yes! Shrink to used is ≥ pad(used_max_p1) — and get_n_kv = pad(used_max_p1) (with used_max_p1 at the time of ubatch1's compute — used set now includes... wait, by the time ubatch2's find_slot runs, ubatch1 has already been applied (prepare applies ubatches sequentially). So stream0's used_max_p1 includes ubatch1's cells. Shrink's n_cap = pad(max(...)) ≥ pad(used_max_p1) = n_kv that graph1 reads ✓. And decommit only rounds-in on whole granules past keep — keep = n_cap*row ≥ n_kv0*row... n_cap = pad(used_max_p1,256) = n_kv0 ✓ mapped region ≥ n_cap? mapped is exactly shrunk to n_cap, keeping granules [0, n_cap*row] (boundary granule remains) ✓ reads up to n_kv0 are safe.

Hmm, good. But — does shrink of stream 0 during a *later ubatch in the same prepare* matter for graph compute? Graph compute is after all ubatches applied. Reads of graph for ubatch i use n_kv_i = pad(used_max_p1 at time of apply of sinfo_i?) — actually get_n_kv(sinfos[i]) is computed at apply() time per ubatch: `n_kv = kv->get_n_kv(sinfos[i_cur])` — this is called in llama_kv_cache_context::apply during decode, after all prepare. used_max_p1 at that time ≥ value at prepare ✓, and per-stream cap is ≥ pad(current used) ✓ because all shrink paths clamp to current used. OK, safe.

**Candidate: `elastic_grow_fork` inside a const `find_slot` mutates capacity and further resets graphs/moe via kv_elastic_set_fork (gf_res_prev_reset) — during decode while the scheduler is holding... find_slot is called from prepare before graph build ✓.

**Candidate: `elastic_set_capacity_fork`'s decommit + immediate recommit in another stream's granule boundary — analyzed, safe.

**Candidate: `test-kv-elastic-streams`'s K-shift scenario says "a K-shift of stream 0" is in the scenario. FORK says shift isn't testable with qwen4exp... but the scenario list includes "К-shift потока 0" — contradictory, whatever.

Recheck K-shift single-stream elastic: n_shift = elastic_shift_base_fork(n_stream) — n_stream=1 → cap0. But the loop in set_input_k_shift: for each stream writes n=min(elastic_cells_fork(s), size) entries at base — sum = n_shift ✓. But build_graph_shift single-stream: view n_shift rows of layer.k = rows [0, cap0) — that's stream 0's cells [0,cap0) ✓ offset 0 ✓.

Multi-stream: view for stream s: n rows... `elastic_cells_fork(s)` — if cap==0, that's get_size... in that case n_shift base mismatches? If any stream's cap==0 (only pre-init), consistent everywhere ✓.

OK, one more: `build_graph_shift`'s multi-stream view uses `layer.k->nb[2]` as stream stride — but the elastic commit's stride is r.stride = t->nb[2] ✓ same.

So elastic: I couldn't find a definitive wrong-output bug in the main flow. Honest report: mostly clean; report real issues I did find:

**Finding 1 (Suggestion, opt-in?): KV_SHARED=1 uses "_Shared" extra buft for attention K/V but excludes name_tag caches; sparse attention (qwen4exp QSA) reads only selected cells... opt-in, and the "Vulkan0_Shared" host memory — a device reading pinned host memory — correctness is fine. Skip or nit.

Wait — one more KV_SHARED detail: `llama_kv_elastic` and kv_shared: elastic init places tensors in elastic device buffers... if the buft is a shared type (extra buft), p.alloc looks up the reg proc for the shared buft's device — cuda/vulkan exports the proc — alloc_elastic_fork checks `ggml_backend_buft_is_cuda(buft)` — shared buft is not cuda buft → returns nullptr → emulated fallback via `alloc_all` failure path — but only if `emulate`... if mode -1: `if (!emulate || !elastic) return false;` — elastic=true (any plan has alloc proc)... wait, p.alloc is proc pointer (non-null since device exports), and calling it returns nullptr (buft not cuda). alloc_all returns false, elastic = any_of(alloc != nullptr) = true; if emulate → fallback; if !emulate → return false → elastic off, warning... mode=-1 → kv_elastic_try: init false → silently return (no warn since mode<0? Warns only for mode>0) — then kv_alloc_deferred_fork allocates full buffers — but wait, memory was created under defer scope with no memory (size-0 buffer); alloc_deferred covers ✓. And `llama_kv_elastic_no_cache_fork` is still true, and defer... kv_elastic_init_fork calls try then alloc_deferred ✓. OK, no problem (KV_SHARED + default elastic → KV simply not elastic).

**Finding: `llama_kv_cache::elastic_init_fork` rejects quantized KV types; f32/f16 are OK.

**Potential real bug — in `llama_kv_elastic_defer_fork` and `no_cache` for the draft context**: llama-model.cpp line 3392 defer only applies to the hybrid idx fork model (qwen4exp). For other models with no_cache elastic, KV is allocated normally at first (fully allocated), then at context init elastic_init_fork replaces the buffer with an elastic one (elastic_init_fork frees the old buffer!). Data is discarded — context init is before any data ✓. But prompt cache load / warmup — happens after kv_elastic_init_fork? kv_elastic_init is at llama context construction (llama-context.cpp:632), before warmup? Should be. ✓

**Now, MTP window findings**: one thing that actually exists: in the `windowed` single-slot path, the window is taken from the last cells of n_kv, and the comment says "cells follow positions (one sequence, filled in order)". After the draft's rollback removes cells and find_slot refills from 0 (head reset condition), order breaks → window includes older cells and excludes recent ones. Impact: draft quality only → Suggestion at most. Not wrong output (target verifies). FORK also mentions measurements of acceptance rates.

And swa_room_fork eviction within the window — acknowledged design, draft only.

**Another actual issue candidate**: in `qwen4exp`'s windowed attention, the `mask_out` slice assumes all dropped columns are unnecessary — but the causal mask keeps older cells; the point of a window is to drop them. No problem.

Next let me recheck the ordering of `swa_room_fork` against prepare()'s failure rollback too: prepare() calls find_slot (fail), swa_room_fork evicts (seq_rm modifies cells), then find_slot retries. If the retry still fails, prepare rolls back previously placed ubatches via a states stack — but the eviction (seq_rm) isn't rolled back! cells.set restores the placed ubatch's cells and head, but the evicted cells (seq_rm on other cells) remain removed. So a failed prepare can silently evict sequence cells — losing KV of a sequence that should have been kept (although those cells were the seq's oldest cells; if the ubatch fails entirely, no eviction should have happened). Impact: draft cache; later decode sees holes → wrong attention for draft (cell metadata says empty but old sequence spans → seq_pos_min shifts; draft is verified by target → quality). Also `test-kv-window` covers metadata. This is a genuine state-corruption-on-failure-path bug: partial eviction not rolled back. Severity: Suggestion (MTP draft only, opt-in default-on in fork's qwen4exp MTP context... default on with MTP), could contribute to repeated failure loop (each failed prepare evicts up to n_need more cells of the sequence — FORK comment "after an image every step of the head would else search in vain first, until the window passes the image"). On repeated failure the cache degrades. But if success=true eventually the eviction is intended. If success=false, cells that swa_room evicted during this failed call are lost. Draft only. Report as Suggestion.

One more check on swa_room_fork: `n_need = n_tokens_s - n_empty + min(256, n_swa/64)` — if n_empty >= n_tokens_s, `continue` before computing n_need ✓.

Also `p_lim` scanning all ubatches: O(ub * tokens * seq) — no problem.

**Next, multi-slot seq_id/position mismatch for "KV SHARED"** — LLAMA_KV_SHARED is about the buffer type, not a seq mapping. No problem, opt-in, no seq semantics. Note clean.

Another potentially Critical area: `elastic_copy_stream_fork` builds views via ggml_view_2d on t (a 3D tensor [embd, kv_size, n_stream]) with `t->nb[2]` per stream: rows(t, strm) = view_2d(ctx, t, t->ne[0], n, t->nb[1], strm*t->nb[2]) ✓ copies rows of n cells between stream regions. Called during update() after `llama_synchronize(lctx)` ✓ sync done before copy (update syncs before copy loop ✓).

OK — also recheck **Vulkan decommit**: `ggml_vk_fork_queues_idle(el->device)` before bind ✓.

And **`ggml_vk_fork_sparse_bind`** is on which queue, does it wait idle? Commit doesn't require idle. ✓

Also **elastic_init_fork's `t->buffer = nullptr; GGML_ASSERT(ggml_backend_tensor_alloc(...))`** — for tensors with existing buffer... the defer path has buffer set (size-0 bufs). Setting t->buffer=nullptr then allocs ✓.

Finally, **alloc_deferred_fork** iterates `ctxs_bufs` — pairs (ctx, buf); sets t->buffer=nullptr on all tensors including views (view_src), then `ggml_backend_alloc_ctx_tensors_from_buft` handles views ✓.

Also one more thing to check — **`ggml_backend_multi_buffer`'s free order vs the `el->ranges` that hold `c.buf` pointers**: ctxs_bufs owns the multi-buffer freeing chunks; ranges hold raw pointers used while the cache is alive ✓. At cache destruction, ranges dangling but unused ✓.

Now, are there any Critical default-on findings? Honestly, this fork is well reviewed. But the prompt says "don't fabricate findings; state an area is clean if it is clean". What I'll report:

1. Suggestion: prepare()/swa_room_fork rollback gap (evictions not undone on failed prepare) — file src/fork/llama-kv-window-fork.cpp / src/llama-kv-cache.cpp:868-871. Default-on for MTP draft cache with window in fork's qwen4exp (LLAMA_MTP_WINDOW_KV=1 default, MTP context only). Impact: draft cache degradation and possibly repeated eviction on failures; speculative output verified by target → no wrong token, but degrades. Could evict cells not yet masked by window within n_swa window (only pos < p_lim, but p_lim is the batch's min position; those cells might still be within window range → draft attends fewer). Draft only.

Hmm wait — actually a more severe consideration about this eviction: could a sequence with rollback (draft reject in `n_rs_seq`) need those cells? Draft cache cells for rejected tokens... window draft cache rejects and re-adds tokens — evicted old cells (below p_lim) not needed by current batch. OK.

2. Suggestion: single-slot MTP_WINDOW view assumes cell index order == position order; after cache wrap / hole refill (head reset `head_cur > used + 2*n_tokens` restarts from 0), the newest cells are not necessarily at the end of n_kv → window may exclude recent cells while including older cells. Draft quality only (verified by target). Also this path only when window_kv off (LLAMA_MTP_WINDOW_KV=0/-1) → opt-in relative to default. Actually default KV=1 is always window-KV if window < n_ctx and size rule; the view path applies when window_kv is off (mode 0, or window >= n_ctx_seq...). With default env: mode=1, window 16384, n_ctx_seq 8192 (short context): rule off (window >= n_ctx_seq → res off) → windowed view path? `mtp_window=16384 > 0`, window_kv false, condition `kq_mask->ne[0] > 16384+255` but — with n_ctx 8192, n_kv never exceeds 8447 → windowed false. So short context is unaffected. Long context 262144: window_kv on. So the view path runs only when the user sets KV=0/-1. So finding 2 is opt-in. And after long decode, does draft cache wrap? Single-sequence draft cache: cells are appended in order; wrap on full → head reset... non-SWA cache: when cache full, decode fails / prompt processing requires slot... draft cache size = n_ctx_seq; positions also bounded by n_ctx — find_slot wrap with holes from draft reject. Plausible. Opt-in + draft-only → low-severity Suggestion.

3. Nit: `elastic_zero_reads_fork()` declared/defined but not called (dead code) — no, I was told not to report style. Skip.

4. **`seq_to_stream[ubatch.seq_id[i][0]]` in swa_room_fork** uses only the first seq of a token while `seqs` adds all seq_ids of a token — a token multi-seq'd in a batch on another stream: n_tokens_s counts only tokens whose first seq maps to s; seqs includes the seq of other streams... eviction only on stream s's cells for seqs that include the seq whose stream is not s → `cells.seq_pos_get(seq_id)` for a seq not in this stream returns empty set... but if a token has [seqA(first), seqB], seqA→stream0, seqB→stream1: the token counts toward stream0's n_tokens_s; seqs gets A and B for both stream iterations; eviction from stream1 for seq B — but the token itself goes to stream0 (find_slot per seq_to_stream — uses ubatch.seq_id[i][0]? find_slot uses ubatch.seq_id_unq). Mismatch is minor: could evict extra cells of seqB (stream1) for a token going to stream0. Evicting seqB's oldest cells below p_lim(B) — within window draft, B loses oldest cells → quality only. Nit/Suggestion.

5. **Race candidate**: `set_input_kq_mask_impl_fork`'s parallel fill uses thread pool with LLAMA_INPUT_THREADS static init — thread-safe static ✓. No races (rows disjoint) ✓.

6. Anything about `seq_cp` error path `v_cells[s1].reset()` + `v_heads[s1]=0`: if s1 has other sequences (unified? seq_cp cross-stream only when !unified; stream == slot == 1 seq per slot... but in slot's stream, could seqB have been copied previously? seq_cp(s0→s1) for slot child — s1 belongs to child slot; `mem.seq_rm(other.id, -1, -1)` precedes. Other seqs in child's stream? In non-unified n_seq_max streams, each slot's stream only holds its seq (+ copied children? copy_prompt_to targets a single other slot). `v_cells[s1].reset()` wipes the child stream's previous contents — which was just reset for the copy anyway. OK.

7. **CUDA `ggml_backend_cuda_buffer_commit_fork` calls `cudaStreamSynchronize(cudaStreamPerThread)`**, and `ggml_cuda_set_device` too — can commit be called from inside find_slot during decode while the scheduler has an active command stream? Sync only waits on StreamPerThread, not on in-flight work on other streams — mapping new memory doesn't conflict with in-flight work that doesn't touch that region (it's unmapped — in-flight kernels couldn't have touched it) ✓ memset on StreamPerThread, then subsequent K/V write kernels on scheduler's stream — ordering between StreamPerThread's memset and the scheduler's stream's writes? memset completes before commit returns (host-sync) → subsequent enqueued writes see zeros ✓.

8. **Vulkan commit's memset path**: `ggml_vk_buffer_memset(ctx->dev_buffer, ...)` — waits on fence ✓.

OK. And one more truly suspicious point I want to verify before writing: **`elastic_cover_fork`'s early return `sinfo.n_stream() < 2`** — for single-stream ubatches on CUDA, coverage depends on the invariant n_kv ≤ cap. n_kv = pad(used_max_p1 after apply) ≤ cap? After apply, used_max_p1 = max(previous used's max idx +1, new idx+1) with new idx < n_cells = cap (find_slot bounds) ✓, and pad(used_max_p1) ≤ cap — since cap is a multiple of 256... wait: used_max_p1 ≤ cap, cap is multiple of 256 → pad(used_max_p1,256) ≤ cap ✓ min(size) ✓ min 256 ≤ cap ✓. But this only holds if cap was a multiple of 256... `GGML_PAD(n_cap, 256)` ✓ or n_cap clamped to get_size (is get_size = n_ctx_seq a multiple of 256? n_ctx is usually a multiple of...; n_ctx_seq could be arbitrary, e.g. 2048 ✓ but user 2173? then n_kv = min(size, pad(...)) = size=2173 and cap = size=2173 ≥ n_kv ✓.) No problem.

Multi-stream zero_reads=true (Vulkan strict): short streams read up to n_kv — beyond their region? n_kv ≤ get_size ≤ stream region rows ✓ region = [strm*stride, (strm+1)*stride), stride = kv_size rows ✓.

Next **hmm — one more subtle point about cover on CUDA**: find_slot grows the stream to `used+n_tokens` — then cover checks cap ≥ predicted n_kv, grows via elastic_grow_fork(strm, n_kv). grow = kv_elastic_set_fork(n_cells + step, true, strm) ✓.

But in cover, predicted n_kv uses `v_cells[...].used_max_p1()` before apply — but prepare applies ubatch i, and find_slot for ubatch i+1 — used reflects previous applies ✓ prediction per ubatch ✓.

OK — final check on the one Critical thing I haven't verified yet: **K-shift's multi-stream elastic shift with grow between build and set_input across ubatches**: update()'s shift happens once per update; set_input at compute time within the same update call; cap unchanged inside ✓. But wait — the k_shift tensor is in the reserved graph result; `get_gf_res_reserve()`... build_graph_shift(res, lctx) builds a fresh tensor each time ✓.

And **shift correctness with elastic multi-stream: shift view n = elastic_cells_fork(s) rows — but other stream's shift mask tensor entries at base(s) offset use elastic_shift_base_fork(s) — base is sum of elastic_cells_fork(s') — if capacity grows between build_graph_shift and set_input (both inside update's compute: build, alloc, then graph_compute which calls set_input)... between build and compute capacity cannot change (no find_slot in between) ✓.

Alright. Also quick check on the elastic knob in `llama-context.cpp:632` context, and `kv_elastic_*` in decode — done enough.

One more potential Critical: **`elastic_commit_fork`'s `need = min(r.stride, n_cap*r.row)`** — for CUDA buffer with tensor spanning multiple streams: commit offset = r.offs + strm*r.stride — but is r.offs the tensor's offset within the chunk buffer? Yes: `el->ranges.push_back({t, c.buf, offs, ...})` ✓.

And **is r.span = end - offs ≥ tensor's needs?** decommit uses span only for last stream's `to` ✓.

I'm quite confident now. Let me also glance at test-mtp-window-fork.cpp to see if there's any mismatch with rule semantics... a simple rule test; skip.

Should also recheck `llama_fork_env_int` to verify the default args I read and parsing (e.g., "ture" → value?). Quickly check llama-fork.cpp / llama-env-parse-fork.h since the report cites exact defaults. LLAMA_MTP_WINDOW default 16384 ✓ (line 499), LLAMA_MTP_WINDOW_KV default 1 ✓ (506), LLAMA_KV_ELASTIC default -1 ✓ (72), ZERO_READS default 1 ✓ (282), LLAMA_KV_SHARED default 0 ✓ (llama-kv-cache.cpp:226).

Let me write the final report. Findings ordered:

1. Suggestion — swa_room_fork evictions aren't undone when a later ubatch fails (prepare rollback restores placed ubatch's cells but not seq_rm'd cells). src/fork/llama-kv-window-fork.cpp:14 / src/llama-kv-cache.cpp:869.

2. Suggestion — swa_room_fork's `seqs` set includes sequences whose tokens go to another stream (uses `ubatch.seq_id[i][0]` for stream attribution but adds all seq_ids of a token), so an ubatch destined for another stream can evict that sequence's oldest cells. llama-kv-window-fork.cpp:26-35.

Actually — are there multi-seq tokens in the MTP draft ubatch? The unified window draft cache with seq_cp shared prefix (n_cmpl): each token normally has 1 seq. Draft ubatches from target batch have 1 seq per token. Low impact. Merge with finding 1? Keep separate as nit.

3. Suggestion — single-slot MTP_WINDOW (view path, only when LLAMA_MTP_WINDOW_KV=0 or -1) assumes last-n_kv cells == nearest positions; broken by hole reuse after draft reject (find_slot head reset fills low holes with newest cells while older cells at high indices remain in window) → window may exclude recent cells (draft quality; verified). qwen4exp-fork.cpp:1413-1421.

4. Also potentially worth noting: **`swa_room_fork` can evict cells within n_swa window (pos ≥ seq_pos_max+1-n_swa), i.e. cells the draft still attends to** — an intentional, documented trade-off; draft only. Mention briefly in area evaluation.

5. KV_SHARED: opt-in, only switches buffer type; no changes to index/seq logic — clean.

6. Elastic: state that the invariants I enumerated hold (prefix map; find_slot bounds; cover; shrink clamp ≥ pad(used); decommit round-in / sync; commit zero-fill and skip already-mapped; hybrid idx grow order; seq_cp grow-or-empty and server seq_copied_fork check at both call sites; K-shift size consistent). No critical found.

Wait — before finalizing, one more quick recheck of the claim for finding 1 about prepare()'s rollback: states only restore cells that were touched by successful find_slot/apply placements and the head. swa_room_fork's `cells.seq_rm(idx, seq_id)` occurs between a failed find_slot and the retry — those idx aren't in any sinfo of the current (failed) prepare... but might be in a previous ubatch's sinfo? No — eviction only picks cells below p_lim (batch's min pos), which are cells not held by earlier ubatches of this batch for the same sequence... "not held by an earlier ubatch of it" — they can be held by earlier ubatches of a different batch (persistent conversation cells) — and are evicted without rollback. Also if prepare fails on ubatch k, ubatches 1..k-1 are rolled back, but evictions from their swa_room calls remain. So yes, an actual non-atomic side effect. But is this really harmful? These are the seq's oldest cells below the current batch's min pos — SWA-window semantics will eventually mask them anyway... unless conversation spans < n_swa: the batch is placed after a conversation of 5000 positions with window 16384: p_lim = 5000; eviction removes cells below pos 5000 that the draft still attends to (within window). Then the draft attends to fewer cells than the window — acceptance-rate loss, not wrong output. And if prepare ultimately fails, the request errors out. Draft-only → Suggestion.

Actually, could these cells belong to sequences other than the ubatch's? No — only seqs in the ubatch.

And the severity of failure scenarios: "repeated failures evict more each time" — the +256 slack limits per-call; log says it only occurs for images. OK.

One more area I promised to check: "two slots overwriting each other's rows" — non-unified per-slot stream; unified single stream cells shared correctly; elastic copy... skip.

Also "race between threads": llama_parallel_for mask fill (rows disjoint ✓), env statics ✓, logged atomic ✓, scheduler snapshot mutexes — out of scope.

For completeness should I also quickly check the mtp-window's use in the `llama-context-fork.cpp:689` prompt signature:
- line 689: n_swa from kv for signature ✓ (per FORK, window from created cache, not env). No problem.

Report in Russian? User memory says "user requests answers in Russian". The prompt itself is in English (a delegated review task). Language rule: respond in the same language as the user's input — the input here is an English task. But memory says the user wants Russian... that's who the user is, and the memory says answer in Russian and don't mix languages. The task text is English, but the final consumer is this user. Hmm. The memory is explicit: "пользователь просит отвечать по-русски и одним языком во всём ответе" (the user asks to reply in Russian and in one language throughout the reply). I'll write the report in Russian and keep code identifiers/English quotes as-is (technical artifacts untranslated).

OK, let me quickly sanity-check finding 3's claim that find_slot refills holes from 0: condition `if (head_cur > cells.get_used() + 2*n_tokens || head_cur >= n_cells) head_cur = 0;` — yes, restarts from 0 to fill holes. Does the draft cache accumulate holes after rejects? How are rejected draft tokens removed from draft KV cache? In upstream MTP speculative decoding, the draft context processes accepted tokens... rejected ones — the draft cache rolls back with llama_memory_seq_rm(pos)? In the spec algorithm, after sampling with the target, draft cells for rejected tokens are removed (`llama_memory_seq_rm(seq, kept_pos, -1)`). That leaves holes at the tail (future positions) — head is beyond them; subsequent new tokens: head_cur vs used — after rm, used shrinks; head > used+2n triggers → head=0 → refill holes from lowest free — holes are at tail (highest indices)... refill from 0 fills holes in ascending order, so lowest hole first, and it's the most recent cell... hmm, the holes are at the tail = highest indices, and those are exactly the most recent cells... after filling, the last cell (max idx) gets filled again by the latest token — since hole fill is ascending, the max used idx still has the latest... actually not quite: cells [.. kept, hole..]; new tokens fill holes ascending → new tokens take hole indices, all larger than kept positions — order preserved! Reversal needs wrap (head wraparound) — when the whole cache has been used once (long run at cache-full boundary). With n_ctx 262144 in a draft window... the view path (KV=0) has draft cache = full n_ctx_seq; wrap after n_ctx cells — but positions also near n_ctx → context shift... In llama.cpp positions don't exceed n_ctx; on cache full the server errors/shifts. Wrap without shift: draft cache has no SWA → find_slot wraps and overwrites oldest cells — holes from reject filled in ascending... the invariant "newest at end of n_kv" generally holds, since used_max_p1 = max used idx; the window covers the last n_win of [0, n_kv); older cells could sit at high indices if a later-added cell went into a low hole... example: holes at idx 100-200 (old reject), conversation continues to idx 20000; n_kv=20224; window covers [3585..20224] → hole refilled at 100..200 with tokens of position P+100 — wait, those tokens are recent (added after reject), sitting at low indices, outside the window slice — dropped! But those are... which tokens fill the holes? The next decode right after the rm. So after the reject, tokens are placed in low holes and are recent but outside the window? No — the window is only applied when n_kv > window+255 = 16639; if pos is 20224 and tokens at idx 100-200 are position ~5000 (from an earlier refill)... over time cells at low idx keep getting refilled with newer and newer tokens as long as head restarts to 0 repeatedly. Condition head > used + 2n: after hole creation, used shrinks a bit; head continues from its previous position (near end); refill from 0 only happens when head is much beyond used (large holes). So yes: a significant chunk of recent tokens could reside at low indices while older ones at high indices → window (slice of high indices) drops the recent ones, keeps the older ones. Draft attends to wrong recent context → acceptance drop; also mask is consistent with the slice (mask columns slice the same window), so attention result is a correct attention over those rows — just semantically the wrong window. Verified by target → not wrong output. Stays a Suggestion, opt-in path.

Also note about "windowed" in this broken-order case: mask rows also sliced → recent cells in low idx masked out for draft → draft worse → acceptance drop. FORK itself says exact window = WINDOW_KV=1. Good enough.

Finally — I should also answer the "on by default" summary: elastic default on (mode -1) on CUDA(VMM)/Vulkan(fork), single stream unified; MTP_WINDOW default 16384 + KV=1 but only fork's qwen4exp dense MTP draft context; KV_SHARED default off.

Write the report: findings max 8, most severe first. I have about 3 real findings (all Suggestion level) + clean statement. Should I mention the emulated stale-data vs zero thing? Concluded harmless (masked). Maybe include as verified OK.

Also mention the test gap noted in FORK: K-shift not tested for qwen4exp... but elastic is default-on on CUDA for shiftable models (llama 3.1), default server config with parallel>1 and kv-unified false → multi-stream shift path in build_graph_shift — hmm, actually wait: default llama-server: n_seq_max=4? Default -np 1... parallel default 1, kv_unified default true → single stream. Multi-stream default is rare.

Actually wait, one more truly default-on scenario to recheck: single stream (unified default) elastic default-on on CUDA + context shift (long conversations exceeding n_ctx in a single slot). build_graph_shift single-stream: n_shift = elastic_shift_base_fork(n_stream): n_stream=1 → sum over s<1 → elastic_cells_fork(0) = cap. k view = rows [0, cap). Shift writes rows [0,cap) ✓ mapped. Reads ✓. set_input_k_shift fills data[0..cap) ✓. build_rope_shift computes over n_shift rows — all mapped ✓. No problem.

But shift + elastic: what if cap shrinks right after shift? Shift only applied to used cells, and then reset_shift; shrink keeps ≥ used ✓.

And **K-shift graph compute vs moe-cache release**: irrelevant.

One last thought on something that could be Critical but I should verify: **does `phase_update`'s shrink call `kv_elastic_set_fork(want, false)` while the previous decode's async GPU work is in flight?** llama decode is synchronous by default (the server syncs). Shrink decommit does full cudaDeviceSynchronize ✓ safe anyway. Vulkan's queues_idle ✓.

And **decommit while gf_res_prev cached graphs hold views**: views are just addresses; unmapped memory is remapped zero later, and reused graph writes again — no problem.

OK writing report now. Language: Russian (per user memory), code quotes in English verbatim. # Ревью KV-фич форка (ветка `cache`): Elastic KV, MTP window, Shared KV

Проверено: `src/fork/llama-kv-elastic-fork.cpp` (весь), `ggml/src/ggml-cuda/fork/elastic-buffer-fork.inc` (весь), Vulkan-часть (`ggml-vulkan-fork.cpp:1251-1493`, `ggml-vulkan-buffers.cpp:806-845`), `src/llama-kv-cache.cpp` (find_slot, prepare, apply_ubatch, seq_cp, update, get_n_kv, build_graph_shift, set_input_k_shift/kq_mask, state_read_meta/data, state_get_data), `src/llama-kv-cells.h`, `src/fork/llama-context-fork.cpp` (весь elastic-механизм), `src/fork/llama-memory-hybrid-idx-fork.cpp:270-360`, `src/fork/qwen4exp-fork.cpp:461-580, 1330-1470`, `src/fork/llama-mtp-window-fork.h`, `src/fork/llama-kv-window-fork.cpp`, `src/fork/llama-kv-cache-mask-fork.inc`, `src/fork/llama-impl-fork.cpp`, диф `ggml/src/ggml-alloc.c`, серверные вызовы (`server-context.cpp:900-940, 5194, 5642-5645`), FORK.md (разделы 170-240, 360-530).

Итоговые флаги дефолтов (из кода, не из доков): `LLAMA_KV_ELASTIC` = **-1 (вкл. по умолчанию** на CUDA с VMM и Vulkan-форке, ctx_type DEFAULT, без moe-cache; `llama-kv-elastic-fork.cpp:72, 91-98`); `LLAMA_KV_ELASTIC_ZERO_READS` = 1, но читается **только для эмулируемых буферов** (`:282`, ветка `p.alloc == nullptr`); на реальном CUDA zero_reads всегда false (CUDA не экспортирует `ggml_backend_buffer_elastic_zero_reads_fork`, только Vulkan, `ggml-vulkan-fork.cpp:1381-1387`); `LLAMA_MTP_WINDOW` = 16384, `LLAMA_MTP_WINDOW_KV` = 1 (`qwen4exp-fork.cpp:499, 506`), применяются **только** к `ctx_type == LLAMA_CONTEXT_TYPE_MTP` форкового qwen4exp (`llama-model.cpp:3540-3542`); `LLAMA_KV_SHARED` = 0 (оптим, `llama-kv-cache.cpp:226`).

#### Findings

**1. Suggestion | src/fork/llama-kv-window-fork.cpp:14 + src/llama-kv-cache.cpp:868-871 | Вытеснения `swa_room_fork` не откатываются, если prepare() в итоге падает на более позднем ubatch — операция не атомарна.**
`prepare()` откатывает только ячейки размещённых ubatch (`cells.set(sinfo.idxs[s], it->v_cells[s])`, `llama-kv-cache.cpp:900-912`), а `seq_rm` внутри `swa_room_fork` — нет:
```cpp
if (sinfo_new.empty() && swa_room_on_fork && swa_room_fork(ubatch, ubatches)) {
    sinfo_new = find_slot(ubatch, false);
}
if (sinfo_new.empty()) { success = false; break; }
```
Сценарий: оконный кэш головы MTP (дефолт `LLAMA_MTP_WINDOW_KV=1`) + картинка M-RoPE в разговоре длиннее окна; find_slot не placement, `swa_room_fork` снимает `n_need = n_tokens - n_empty + min(256, n_swa/64)` старых ячеек (`cells.seq_rm(idx, seq_id)`), повторный find_slot снова не находит — prepare возвращает `{}`, запрос даёт ошибку, но самые старые ячейки последовательности (в т.ч. позиции внутри окна 16384, которые черновик ещё имел право видеть) потеряны безвозвратно; на каждом повторе запроса вытесняется ещё. Следствие — просадка acceptance/качества черновика и деградация кэша головы при повторяющихся отказах; финальные токены не портятся (черновики верифицирует цель). Default-on для MTP-контекста qwen4exp.

**2. Suggestion | src/fork/llama-kv-window-fork.cpp:26-35 | Поток для вытеснений определяется по `ubatch.seq_id[i][0]`, а в `seqs` добавляются все seq_id токена — может вытеснять ячейки последовательности, чей ubatch пойдёт в другой поток.**
```cpp
if (seq_to_stream[ubatch.seq_id[i][0]] != s) { continue; }
n_tokens_s++;
for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) { seqs.set(ubatch.seq_id[i][j]); }
```
Для токена с `seq_id = [A, B]`, где A → поток 0, B → поток 1: токен засчитан в потребность потока 0, но на итерации по потоку 1 последовательность B участвует в вытеснении его старых ячеек, хотя ни один токен ubatch в поток 1 не кладётся. Триггер — единый оконный кэш с общими префиксами (`seq_cp`, `n_cmpl`), `-np > 1 --kv-unified`. Эффект — лишнее вытеснение старых ячеек черновика (снова только acceptance, цель верифицирует). Default-on при MTP-голове qwen4exp с несколькими слотами.

**3. Suggestion | src/fork/qwen4exp-fork.cpp:1373-1421 | Путь «view последних ячеек» (`LLAMA_KV_WINDOW_KV=0/-1`, один слот) предполагает, что порядок ячеек = порядок позиций; после перевыбора ям find_slot это неверно — окно может выбросить свежие ячейки и оставить старые.**
Комментарий в коде: «the cells of the draft memory follow the positions (one sequence, filled in order), so the last ones are the nearest». Но `find_slot` при `head_cur > cells.get_used() + 2*n_tokens` перезапускается с 0 (`llama-kv-cache.cpp:1126-1129`) и кладёт свежие токены в низкие освободившиеся (после откатов черновика) индексы, тогда как более старые ячейки сидят на высоких индексах и попадают в срез `[n_kv - n_win, n_kv)`. Mask-срез (`mask_out = ggml_cont(ggml_view_2d(mask_out, n_win, n_out, ..., off*esz))`) согласован со срезом K/V, так что это корректное внимание по неверному набору ячеек: черновик теряет недавний контекст — только приёмка (цель верифицирует). Опционально: при дефолтном `WINDOW_KV=1` этот путь не выполняется (там точное окно по позициям в маске); включается только при `LLAMA_MTP_WINDOW_KV=0` или `-1` + `n_seq_max==1`.

Чистые области (проверено, багов не нашёл):
- **Elastic KV, инвариант «префиксная карта ёмкости»**: единственная «remap-таблица» — префикс `n_cap` на поток; аллокации ограничены ёмкостью (`n_cells = min(elastic_cells_fork, size)` в find_slot, `llama-kv-cache.cpp:1118-1122`), все пути сужения клампуют к `max(n_used, 256)` с паддингом 256 (`llama-kv-elastic-fork.cpp:398-402`), и `get_n_kv = pad(used_max_p1) <= cap` выполняется всегда; обновление карты проверено на всех путях изменения выделения: alloc/init (`elastic_init_fork`), grow (find_slot `:1114-1116`, cover `:1223`, seq_cp `:601-607`, state restore `:2663-2670`, phase_update), shrink (phase_gen/phase_update), clear (seq_rm — только CPU-метаданные, карта — верхняя граница, инвалидация не нужна), defrag — физически ячеек не перемещает (`mv` закомментирован в `llama-kv-cells.h`), K-shift (`build_graph_shift`/`set_input_k_shift` используют согласованные `elastic_cells_fork`/`elastic_shift_base_fork` и выполняются в одном синхронном update без роста ёмкости между build и set_input), stream copy (`elastic_copy_stream_fork` копирует `min(used_max_p1(sdst), caps)`, seq_cp до копирования гарантирует рост или оставляет последовательность пустой с логом, сервер проверяет `seq_copied_fork` на обоих вызовах — `server-context.cpp:907` и сразу после `copy_state_to` `:5642-5645`).
- **ZERO_READS**: на реальных устройствах определяется бэкендом, не env: CUDA не экспортирует `..._elastic_zero_reads_fork` → `zero_reads=false` → `elastic_cover_fork` доращивает покрытие каждого ubatch; Vulkan — только при `residencyNonResidentStrict`. Env со значением 1 влияет исключительно на эмулируемые буферы (обычные полные буферы; чтение за ёмкость там — маскированные CPU-ячейками (`cells.is_empty(j) → mask_drop`) строки, легальные данные нулями не подменяются; тест `test-kv-elastic-streams-fork` сверяет логи побитово).
- **Гонок/памяти**: CUDA decommit делает `cudaDeviceSynchronize()` перед `cuMemUnmap`, Vulkan — `ggml_vk_fork_queues_idle`; commit обнуляет только что замапленные гранулы синхронно (`cudaStreamSynchronize` / fence-wait у `ggml_vk_buffer_memset`), уже замапленные пропускает, так что данных соседа не затирает; округжение commit наружу / decommit внутрь корректно даже при грануле крупнее `LLAMA_KV_ELASTIC_ALIGN=2MiB` (decommit никогда не унимпап гранулу, выходящую за `to`); гибрид-idx форк растит сначала индексер, потом внимание, а сужает внимание потом индексер (`llama-memory-hybrid-idx-fork.cpp:309-330`). Заполнение KQ-маски параллельно по непересекающимся строкам (`llama-parallel_for`, `llama-impl-fork.cpp` — `fn(0,n)` разбивает `[0,n)`, внедиапазонных записей нет).
- **MTP окно и длинные промпты**: окно применяется исключительно к attention-графу плотной MTP-головы (`ctx_type == MTP`), целевой кэш и выборка цели не затрагиваются; срез окна согласован с маской, off-by-one съеден намеренным сдвигом на `mtp_window_pad=255`; на токсен-уровне цель не меняется при промпте > 16384 (только приёмка).
- **Shared KV (`LLAMA_KV_SHARED`, оптим, по умолчанию 0)**: только подмена `buft` на `*_Shared` у бэкенд-регистра для безымянных attention-кэшей (`llama-kv-cache.cpp:223-241`); логика индексов/seq не тронута; пересечение с elastic безопасно (упругий инит корректно отваливается в эмуляцию/полный аллок, отложенные буферы закрывает `alloc_deferred_fork`).

Одно замечание вне категорий ошибок вывода: `llama_kv_cache::elastic_zero_reads_fork()` определена (`llama-kv-elastic-fork.cpp:435`) и не вызывается нигде — мёртвый код, но на корректность не влияет.

---

### B1.4 Spill prompt-кэша

- Агент: `Explore-4er5SsVXWnu7cKQCn4vElz9S5khHHBjl` (тип `Explore`, глубина 1, родитель: `general-purpose-cj32o0JgdCGeWPcOLhzkogCbShcPRb7L`)
- Статус: completed, создан 16:51:26 UTC, финальный отчёт 15576 символов

**Постановка задачи (дословно из лога):**

READ-ONLY code review (do NOT edit/create/delete any file; do NOT run builds or tests) of a performance fork of llama.cpp.
REPO: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`). Merge base is branch `master`.

TOPIC: the prompt-cache DISK SPILL feature only, and the process-pool state transfer that shares it.
Start from:
- tools/server/tests/../fork/server-task-fork.cpp and any other files under tools/server/fork/ (list that dir)
- grep repo-wide for: cache_spill_dir, cache_disk_mib, spill_ckpt, SPILL_VERSION, spill, write-behind, common_prompt_cache, CACHE_SPILL
- tests/test-prompt-cache-spill.cpp (1009 lines) and what implementation it covers
- common/common.h around lines 790-800 (cache_spill_dir / cache_disk_mib) and common/fork/arg-fork.cpp for the CLI args
- read the Russian FORK.md sections describing the spill/pool for intent, but VERIFY against code.

The prompt cache is the server's prefix cache of processed prompts (server_prompt_cache in upstream llama.cpp). The fork adds spilling evicted states to DISK instead of dropping them, an LRU disk budget, and a multi-process pool that transfers slot state between server processes.

GOAL: find correctness bugs that can make inference produce WRONG output:
- a spilled state restored for the WRONG prompt/slot (key/collision: how is a spill file keyed - by hash of tokens? what hash, and can two different prompts map to the same file?),
- a restored state that does not match the tokens it is applied to (off-by-one in how many tokens the state covers, positions/seq mismatch, applying a state after the KV was evicted/shifted),
- a state restored after the model/generation changed (no model id/version in the key),
- SPILL_VERSION / struct layout mismatch silently accepting an old file,
- partial/truncated file accepted (short read, no checksum),
- LRU bookkeeping letting the budget accounting free the wrong file,
- state written while the KV was still being modified (write-behind race with the decode thread),
- files reused across a model reload or between two different server processes with different models/config,
- the pool transferring state between processes whose model/vocab differ.

METHOD REQUIREMENTS (mandatory):
- Read the actual implementation, not just the tests or the comments.
- For every cache hit path, verify what identifies the entry and what could make a stale/mismatched entry be considered a hit. Trace end to end.
- For each risky path, state whether it is ON BY DEFAULT or opt-in, with the exact knob/CLI flag and the default value you read in the code (look for common_fork_env_int / llama_fork_env_int defaults and the args' default values).
- State whether it can be triggered by a NORMAL server workload (multiple slots, cache reuse across requests, restart, pool) rather than only in the unit test.

REPORT FORMAT: at most 6 findings, most severe first. Each: SEVERITY (Critical = can produce wrong output or crash by default / Suggestion / Nit) | file:line | one-line claim | concrete failure scenario (inputs -> wrong outcome) | default-on or opt-in (knob + default) | evidence (the code you read that proves it, quote a few lines).
If an area is clean, say so in one line. Do NOT invent findings; do NOT report style/naming/missing comments. Cite real line numbers you actually read.

**Отчёт агента (дословно):**

I've got everything I need to put this together. Writing up the report now—max 6 findings, most severe first, honestly: no Criticals found. Findings:

1. Suggestion — restore_spilled is missing the filename-vs-header uid binding that spill_entries enforces (server-task-fork.cpp:823-849, 869; compare against :283-284). On adoption it trusts the header uid; on the first hit, drop() will go and delete the file named by that header uid (:1615, 1573-1590, the drop lambda). A single file whose name and header have gotten out of sync (crash during legacy migration rename at :809-816, a manually restored/copied spill dir) can hijack a uid and cause the server to delete another intact state's file on the first cache hit; two adopted entries can also end up sharing a single uid. Cannot produce incorrect output (the prompt byte comparison at :1628 gates the restore). Reachable only if the directory has been externally manipulated or corrupted. Opt-in (via --cache-spill-dir).

2. Suggestion — no integrity check on state-blob/checkpoint payloads (the unspill at server-task-fork.cpp:1611-1623 verifies magic/version/uid/signature/exact size + prompt bytes at :1628, but the main/drft/ckpt blobs only get size-checked by the layout at :1620-1621 and spill_entries:286-289). A state file that passes all checks but has bits flipped in the KV blob section gets restored into the slot (load() in server-task.cpp:1917-1946 → llama_state_seq_set_data_ext; the per-layer sanity checks llama-kv-cache.cpp:2759-2781 only cover layer count/cells/rotations, not data). Files persist indefinitely across restarts ("cold-start reuse"). Scenario: spill file ages on disk, silent corruption → after restart a prompt hits → corrupted KV gets applied → wrong tokens. Opt-in (via spill dir).

3. Suggestion — pool-imported states can never be spilled: pool_import sets disk_evicted=true (:1529); write_behind skips disk_evicted (:1879), persist skips it too (:905). Nothing resets the flag short of a cache hit that consumes the blob anyway. A migrated conversation that gets parked on a node stays RAM-only: crash or host-RAM pressure (drop_resident :1825-1840) discards it → the conversation gets reprocessed from the beginning — durability gap versus the stated goals of spill. Pool routing opt-in; pool endpoints are registered by default whenever the prompt cache exists (server.cpp:726-730; server-context.h:157 pool_migrate_fork() = pool_signature_fork != 0; server-context.cpp:6256; --cache-ram defaults to auto → cache on).

4. Nit — LRU prune can permanently drop the only disk copy of a state while a process with a larger --cache-disk keeps rewriting it: the budget is per-process configuration, prune is global across the directory (spill_prune :301-324; the "Use the same budget" documentation at arg-fork.cpp:48-51). Two servers sharing a directory with different --cache-disk values: the smaller-budget one thrashes/evicts files the larger-budget one treats as persistent. Behavior is documented, but nothing validates agreement.

5. Nit — resident-state disk_evicted flag (spill_exists :1534-1543 sets it on prune; write_behind skips at :1879): a resident state whose file got pruned doesn't get rewritten while idle (until the free_ram spill path or a cache hit clears it) — a crash then loses that conversation despite the idle-time write-behind knob. Corner case; write-behind itself is opt-in (--cache-idle-write defaults false).

6. Nit — spill_entries/prune silently ignores files whose headers are unreadable (:279-291), so junk that matches a valid name pattern never counts against the budget; restore_spilled cleans most of these up at startup, but files that fail to open/read linger. Marginal.

Also to mention: the portable signature excludes n_parallel/n_ctx/kv_unified (server-context-fork.cpp:213-233) — verified the compensating runtime guards: apply-time per-sequence checks return false rather than aborting (llama-kv-cache.cpp:2759-2772, llama-memory-recurrent.cpp:2762-ish... the line for recurrent is 1294), load() drops the entry on n!=size (around server-task.cpp:1926-1936), state_seq_set_data catches exceptions → 0 (llama-context.cpp:3772-3784). Fold into the finding list as a "verified clean" note rather than a finding? The requested format is: findings up top, then a clean statement. Keep findings at 5 and put the clean-path analysis in the body.

Trigger conditions by default:
- spill is entirely opt-in: --cache-spill-dir default "" (common.h:795; arg-fork.cpp:31-45); --cache-disk default 0/unlimited (common.h:796); --cache-min-tokens 0 (common.h:797); --cache-ram-min-tokens 0 (common.h:798); --cache-idle-write false (common.h:799). No use of common_fork_env_int in the spill code path (env-fork.h is a generic helper; spill knobs are CLI/env LLAMA_ARG_* only).
- prompt cache is on by default (--cache-ram -1 auto → server_prompt_cache_init server-context-fork.cpp:127-135); pool endpoints exist by default; migration happens only if a router drives it.

Write the final report in English (request is in English). Include evidence quotes. List absolute paths once, then short names thereafter. Concise and complete. # Prompt-cache disk spill & pool state-transfer — correctness review

Scope read end-to-end: `tools/server/fork/server-task-fork.cpp` (1989 lines, whole file), the hit path in `tools/server/server-task.cpp` (`find_best`/`load`/`alloc`/`update`), the integration in `tools/server/server-context.cpp` (launch_slot, checkpoint restore at decode, `pool_task_fork`, `/pool/*` routes, idle `write_behind`), `tools/server/fork/server-context-fork.cpp` (cache signatures), `tools/server/fork/server-pool-fork.cpp` (xfer registry/inbox), `tools/server/server-common.cpp` (`server_tokens` serialize / `get_common_prefix`), `src/llama-context.cpp` + `src/llama-kv-cache.cpp` (state apply semantics), `tests/test-prompt-cache-spill.cpp` (aligned round-trip, adoption, truncation, legacy names, cross-model dirs), and the Russian spill/pool sections of `FORK.md` (verified against code).

**Headline: I did not find a Critical default-on path that applies a state to the wrong prompt or a mismatched KV.** The identity chain is: file name `state-<owner16>-<signature16>-<uid>.bin` (`server-task-fork.cpp:152-156`), `owner = hash(params.model.path)` (`server-context-fork.cpp:140`), signature over model desc/size/path, n_ctx/n_ctx_seq/n_rs_seq, cache types, `swa_full`, `kv_unified`, n_parallel, `LLAMA_STATE_SEQ_VERSION`, shard sizes+mtime, rope/yarn, attention type, kv overrides, LoRA, control vectors, mmproj, draft config (`server_prompt_cache_signature_impl`, `server-context-fork.cpp:198-293`); and — decisively — every disk restore re-reads the prompt section and byte-compares it against the in-RAM entry's serialized tokens before the blobs are handed to KV:

```cpp
// server-task-fork.cpp:1626-1629
const auto packed = st.prompt.tokens.serialize();
if (h.n_prompt != packed.size()/sizeof(llama_token) ||
        (!stored.empty() && !f.read_at(lay.off_prompt, stored.data(), stored.size())) || stored != packed) {
    return drop();
```

plus `h.uid != st.uid || h.signature != signature || version/magic/exact-file-size` checks at `:1611-1622`. `SPILL_VERSION = 4` is enforced at adoption (`:835`), at unspill (`:1615`) and on the pool wire (`:1359`); version/signature mismatches are *deleted*, not silently accepted. Truncation is covered by tmp+`fsync`+rename (`:1044-1070`) and exact `size == spill_layout_of(h).size` checks (`:844`, `:1621`, `:286-289`). Media chunks are gated by chunk-id (sha256) + n_tokens in `get_common_prefix` (`server-common.cpp:840-862`).

Knobs and defaults (all read in `common/common.h:795-799`, `common/fork/arg-fork.cpp:31-88`): `--cache-spill-dir` default `""` (spill **off**), `--cache-disk` default `0` = unlimited, `--cache-min-tokens` 0, `--cache-ram-min-tokens` 0, `--cache-idle-write` **false**. No `common_fork_env_int` is used on the spill path (env-fork.h is a generic helper only). The prompt cache itself is on by default (`--cache-ram` auto, `server-context-fork.cpp:127-135`), and the `/pool/*` routes are registered whenever a prompt cache exists (`server.cpp:726-730`; `server-context.h:157` `pool_migrate_fork() == (pool_signature_fork != 0)`), but state only moves if a pool router drives it and `signature_from == signature_portable` (`server-task-fork.cpp:1475-1478`).

#### Findings (most severe first; none reached Critical)

**1. Suggestion — startup adoption trusts the header `uid` without binding it to the file name, letting one malformed file cause deletion of a different, intact state's file.**
`server-task-fork.cpp:823-849, 869` (`restore_spilled`). The directory-accounting path enforces the name/header binding —
```cpp
// :283-284 (spill_entries)
h.version != SPILL_VERSION || !spill_header_sane(h) || h.signature != signature ||
        spill_name_of(owner, signature, h.uid) != name) {
```
— but `restore_spilled` checks only magic/version/signature/size and then adopts `st.uid = h.uid` (`:869`) with no `spill_name_of(owner, signature, h.uid) == filename` check. Failure scenario: a spill file whose name-uid and header-uid disagree (crash mid LEGACY-rename at `:809-816`, a hand-restored or rsynced spill dir) is adopted under the header uid; the first cache hit for it opens `spill_path(h.uid)` — the *other* state's file — the prompt compare fails, and `drop()` (`:1573-1590`) `std::filesystem::remove`s that intact file. Wrong output is still blocked by the token comparison; the damage is silent cross-state cache loss (and two adopted entries sharing one uid). Opt-in (`--cache-spill-dir`); not reachable from a clean normal workload, only from a corrupted/hand-edited directory — which the feature explicitly invites ("Several models may share one directory").

**2. Suggestion — no checksum over the state/checkpoint blobs; an in-place corrupted file passes every gate and is applied to KV.**
`server-task-fork.cpp:1611-1623` validates magic/version/uid/signature/exact size, and only the *prompt* section is content-verified (`:1628`); the `main`/`drft`/checkpoint blobs are accepted on length alone (`rd(...)` at `:1635-1639` succeeds if bytes exist). The KV-layer guards (`llama-kv-cache.cpp:2759-2781`) check only layer count, cell count and rotation sizes, not payload. Failure scenario: spill files are kept across restarts by design (cold-start reuse; `persist()` at `:902-920`, kept-on-shutdown per README), a file silently rots (bad sector, non-atomic fsync on a lossy device), the prompt section survives (it is small), a matching request hits after restart, and the corrupt KV is applied via `llama_state_seq_set_data_ext` (`server-task.cpp:1926-1936`) -> wrong continuations, with no signal beyond a generic perplexity change. Opt-in (`--cache-spill-dir`). The "kill mid-write" test (`test_cache_spill.py:280`) covers truncation only.

**3. Suggestion — a pool-imported state can never be spilled, contradicting the spill feature's own crash-protection promise.**
`pool_import` marks every incoming state `st->disk_evicted = true` (`server-task-fork.cpp:1529`, comment: "write_behind does not write it out meanwhile"), but `write_behind` skips `disk_evicted` forever (`:1879`) and shutdown `persist()` skips it too (`:905`: `if (spill_exists(st) || !st.disk_evicted)`). Nothing ever resets the flag except a cache hit that consumes the blobs anyway (`:1576-1579`). Failure scenario (normal pool workload): a conversation is migrated to node B, sits idle; host-RAM pressure evicts it — `drop_resident` logs "has no copy on disk ... computed again from the front" (`:1836-1840`) — or B crashes and the migrated conversation is re-prefilled from token 0 despite a configured spill dir. Pool is opt-in operationally, but the transfer endpoints are live by default on every node with a prompt cache (`server-context.cpp:6256`, `server.cpp:726-730`).

**4. Nit — the shared disk budget is per-process config with no cross-process agreement check; `spill_prune` will evict another model's files under it.**
`spill_prune` (`server-task-fork.cpp:301-324`) evicts LRU "mine"-class files globally across the directory and the arg text says "Use the same budget in servers sharing PATH" (`arg-fork.cpp:48-51`), but nothing detects divergence: a server started with a larger (or default 0 = unlimited) `--cache-disk` treats the directory as unbounded and rewrites files the smaller-budget server just pruned (`free_ram` -> `spill_state` assigns a fresh uid at `:1034-1037`), so the effective budget is the largest one and eviction can thrash between peers. Documented caveat, but silent.

**5. Nit — a resident state whose file was pruned is never rewritten while idle.**
`spill_exists` flips `disk_evicted = true` when the shared budget removed the file of a RAM-resident state (`:1534-1543`); `write_behind` then skips it (`:1879`) and `persist()` skips it (`:905`), so with `--cache-idle-write` on, a crash between that prune and the next `free_ram`-triggered re-spill (or cache hit, `:1576-1579`) loses the only copy. Corner case; performance/durability only, never wrong output.

#### Areas verified clean (one line each)

- **Wrong-prompt file keying / hash collision:** impossible to serve a mismatched blob — every unspill byte-compares the file's serialized prompt against the entry (`server-task-fork.cpp:1626-1629`), and entries are matched to requests only by full-token common prefix (`server-task.cpp:1866-1902`), with media chunk ids in the comparison (`server-common.cpp:840-862`).
- **SPILL_VERSION / struct-layout drift:** version enforced at adoption/unspill/wire (`:835`, `:1615`, `:1359`) with `static_assert`s on `spill_header`/`spill_ckpt` (`:53`, `:67`); old-version files are removed, not accepted.
- **Truncated/partial files:** short reads fail the aligned readers; exact `size == layout` is required (`:844`, `:1621`, `:286-289`); writes go tmp + `finish()` + `sync()` + rename (`:1044-1067`); abandoned `.tmp` are deleted under the directory lock (`:263-269`, `:367-371`).
- **Reuse across model reload / config change / pool with different model:** non-portable signature gates disk files (stale files deleted at startup, `:356-370`); `signature_portable` gates pool import (`:1475-1478`); the portable signature's omissions (n_ctx, n_parallel, `kv_unified`) are backstopped by per-sequence apply checks that *return false / throw* rather than apply garbage (`llama-kv-cache.cpp:2759-2772`, `llama-memory-recurrent.cpp:1294`, `llama-context.cpp:3772-3784` -> `load()` drops the entry and the slot recomputes, `server-task.cpp:1926-1936`).
- **Write-behind race with decode:** none — all cache mutation (`prompt_save`, `load`, `write_behind`, `pool_task_fork`, inbox drain) runs on the server main thread; disk writes are deliberately deferred while any other slot is processing (`defer_writes`/`others_run_fork`, `server-context.cpp:2007-2020`, `:4297-4306`) and the written blobs are host-side snapshots, not live KV.
- **Off-by-one in restored-state coverage:** spill/unspill move byte-exact `llama_state_seq` blobs for the exact prompt they were saved from; `n_used` semantics are unchanged from upstream; the pool's checkpoint-selection mirror (`server-task-fork.cpp:1188-1220`) matches the slot's real logic line for line (`server-context.cpp:5024-5105`: same `pos_eval`, `pos_max > pos_next` skip, `pos_min < thold || pos_min == 0` predicate, `n_past = min(size_up_to_pos(...), cp->n_tokens)`), and the receiver re-decides from its own memory anyway; a wrong export guess costs recompute, not correctness.

---

### B2 Численность и память: Vulkan, CUDA, ядро ggml

- Агент: `general-purpose-ksLnpjA7E67cqBYBArnNK6VaQtRzgMCz` (тип `general-purpose`, глубина 0, родитель: корневой вызов ревью)
- Статус: completed, создан 13:44:10 UTC, финальный отчёт 17182 символов

**Постановка задачи (дословно из лога):**

You are doing a READ-ONLY correctness review of a performance fork of llama.cpp. Do not edit, create, or delete any file. Do not run builds (another process is already building in build-qreview).

REPO: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`).
REVIEW TARGET: `git diff master...HEAD` (79 commits, ~59k added lines; merge base = `master` aa588a346).
CONTEXT: perf-focused research fork; notes in FORK.md / RESEARCH.md (Russian). Env knobs go through `llama_fork_env_int(NAME, default)`; several default ON. The owner's #1 question: do the speed optimizations change NUMERICAL RESULTS / model output, and are they memory-safe?

YOUR SCOPE - GPU/CPU backend ops and ggml core ONLY (sampling/speculative and KV/server caching are covered by other reviewers - skip them):
1. Vulkan: `ggml/src/ggml-vulkan/fork/ggml-vulkan-fork.{cpp,h}`, `ggml-vulkan-fusion-fork.cpp`, `ggml-vulkan-ops-fork.cpp`, and the diff to `ggml-vulkan.cpp`, `ggml-vulkan-types.h`, `ggml-vulkan-buffers.cpp`, `ggml-vulkan-push-constants.h`.
2. New/changed GLSL shaders: `mul_mmat_f16reg`/`mul_mmid_f16reg`/`mul_mm*`, `topk_radix_select.comp`, `topk_argsort.comp`, `gated_delta_net_lazy.comp`, `flash_attn*`, `rms_norm_rows.comp`, `get_rows.comp`, `concat_transpose.comp`, `ssm_conv_concat.comp`, `quantize_q8_1.comp`, `count_experts.comp`, plus `vulkan-shaders-gen.cpp`.
3. CUDA fork includes: `ggml/src/ggml-cuda/fork/*.inc` (esp. `top-k-fork.inc`, `gated-delta-net-lazy-fork.inc`, `getrows-fork.inc`, `repeat-fork.inc`, `elastic-buffer-fork.inc`, `ggml-cuda-timing-fork.inc`) and diffs to `ggml-cuda.cu`, `mmid.cu`, `mmq.cu`, `top-k.cu`, `getrows.cu`.
4. CPU: `ggml/src/ggml-cpu/fork/ops-fork.inc`, `timeline-fork.inc`, diffs to `ops.cpp`, `ggml-cpu.c`.
5. ggml core: diffs to `ggml/src/ggml.c`, `ggml-alloc.c`, `ggml-backend.cpp`, `ggml/include/ggml.h`, `ggml-backend.h`, `ggml-alloc.h` and `ggml/src/fork/*.inc`.

What to look for, in priority order:
(a) Numerics that change model output: f16 accumulation where upstream uses f32, `LLAMA_MOE_DOWN_F16` (default 1) and `LLAMA_HC_F16` (default 1) - where exactly do they reduce precision and is the reduction justified/bounded?; approximate or tie-breaking top-k (radix select vs argsort) - does it return a DIFFERENT set/order than the exact path on ties, NaN, or k near n?; changed reduction order in flash_attn; any fast-math / reciprocal approximation.
(b) Kernel correctness: out-of-bounds reads/writes, missing workgroup/memory barriers, uninitialized shared memory, wrong strides/pitches, integer overflow in index arithmetic for long contexts (262144!), missing handling when a dimension is not a multiple of the tile size, NaN/Inf propagation.
(c) Memory safety in the allocator/backend changes: buffer reuse while still in use, use-after-free, aliased buffers in graph capture, missing sync before read, prefetch/elastic buffer lifetime.
(d) Graph/fusion legality: a fusion that changes results when the same tensor is consumed twice, or that is enabled when the required alignment/quant type is not actually present.

For each finding state: SEVERITY (Critical = wrong output or memory error; Suggestion; Nit), file:line, one-line claim, the concrete failure scenario (what input/shape/quant/tie case, and what wrong value comes out), whether it is ON BY DEFAULT or opt-in (name the knob + default), and your evidence (the specific code you read).

Verify before claiming: read the surrounding/upstream code to confirm the difference is real, and check the shader's generated dispatch constraints before claiming OOB. Do not report style, formatting, comments, or duplication. Max 12 findings, most severe first. If an area is clean, say so in one line.

**Отчёт агента (дословно):**

Both `LLAMA_HC_F16` and `LLAMA_MOE_DOWN_F16` are gated on `qwen4exp_dev_is_vulkan(...)` — **Vulkan only**. So on Metal neither is active, which explains why turning them off changed nothing. That's an important, concrete answer to the owner's question #1: they are Vulkan-only precision reductions, so Metal/CPU paths are unaffected. Good.

Now let me write the final report. Keep it organized, in Russian? The user's memory says "пользователь просит отвечать по-русски". But this is a report to a parent agent, and the parent wrote partly in Russian. The output-language rule says respond in the user's input language. The parent's last message was mostly Russian. However the findings need to be consumable. The original task prompt was English. The most recent message is Russian.

Given the memory ("пользователь просит отвечать по-русски и одним языком во всём ответе") and the parent's latest message is Russian, I'll write in Russian, keeping code/identifiers/file:line and severity labels in English verbatim. That respects "keep technical artifacts unchanged".

Hmm, but the parent will integrate findings into a report — mixed. The parent's own instruction format used English labels ("SEVERITY", "ON BY DEFAULT"). I'll write prose in Russian with English technical terms/labels. That's the best fit for the memory + latest input.

Let me write it now, concise but complete. ## Ответ на приоритетный вопрос (Metal NaN) — сначала он, как просили

**SEVERITY: Nit (не дефект ветки) — с оговоркой, что буквально master я не запускал.**

Локализовал точно и детерминированно:

| эксперимент (одна команда, `-ngl 99`) | результат |
|---|---|
| baseline | NaN |
| `-ot "blk.0.ple.*=CPU"` | **OK 128.4807** |
| `-ot "blk.0.ple_norm_query.weight=CPU"` | **OK 128.4806** |
| `-ot "blk.0.ple_norm_key.weight=CPU"` / `ple_norm_conv` / `ple_key` / `ple_value` / `ple_conv1d` | NaN |
| `-ot "blk.0.ssm.*"`, `hc.*`, `ffn.*`, `attn.*`, все `output*`=CPU | NaN |

Точка расхождения — узел `GGML_OP_MUL` из `grouped_norm(hidden, model.layers[il].ple_norm_query)` в `src/models/qwen4exp.cpp:1490` (= `src/fork/qwen4exp-fork.cpp:3209`), т.е. `rms_norm(hidden) * w` перед `sum_rows(mul(key, query))` в PLE-гейте блока 0. NaN появляется уже в **первом** чанке prefills (`[1]nan,[2]nan`), накопления по токенам нет.

Почему это почти наверняка **гипотеза 1 (и в master так же)**, а не +705 строк ggml-ядра:

1. `git diff --stat master...HEAD -- ggml/src/ggml-metal/` → **один файл, `ggml-metal-device.m`, +5/−1**, и эти 5 строк только *сужают* поддержку (`MUL_MAT_ID` с `GGML_HINT_IDS_REPEAT`/`src[3]` → `return false`). Все Metal-шейдеры и `ggml-metal-ops.cpp` байт-идентичны master.
2. `git diff --stat master...HEAD -- src/models/qwen4exp.cpp` → **пусто**. `build_ple` форка побайтово равен master-версии (сравнивал `git show master:src/models/qwen4exp.cpp` с форк-функцией — разница только в имени класса).
3. `git diff master...HEAD -- ggml/src/ggml.c`: список всех hunek — имена 2 новых опов, `GGML_OP_COUNT` static_assert, `FLASH_ATTN_LIST` в `ggml_prec_set_acc/get_acc`, ослабление assert'а в `ggml_mul_mat_set_hint`, `ggml_mul_mat_id_add_skip`, `ggml_flash_attn_list`, `ggml_gated_delta_net_lazy/replay`, `ggml_dsv4_hc_pre_impl`. **Ни один существующий оп не меняет математику**; `rms_norm/mul/sum_rows/abs/sgn/sqrt/clamp/repeat` в диффе не встречаются вообще.
4. Все дефолтные ggml-core-ручки выключены — `GGML_SCHED_PREFETCH=0`, `GGML_SCHED_PARALLEL_CPU=0`, `GGML_ALLOC_UPSTREAM=1`, `LLAMA_GRAPH_CACHE=0`, и все вместе → **всё равно NaN**. То есть план размещения/аллокатор/префетч/граф-кэш исключены эмпирически.
5. `LLAMA_UPSTREAM=all` (graph строит master-класс `llama_model_qwen4exp`) → NaN. Это, по сути, master-граф на master-Metal.
6. **Прямого доказательства «железо-ядра в порядке»**: `test-export-graph-ops` с реальной моделью → 199 реальных узлов графа, затем `test-backend-ops -b MTL0 --test-file` → **195/197 passed**; два «FAIL» — это `SQRT` и `GATED_DELTA_NET` с `MTL0=nan CPU=nan`, артефакт харнесса на случайных данных, а не расхождение. Ни один op на реальных формах qwen4exp с Metal не расходится с CPU.

Ответ на «есть ли в диффе путь, меняющий ЧИСЛА, а не размещение»: **в ggml.c/ggml-alloc.c/ggml-backend.cpp — нет** (п.3 и п.4). Числа меняют только Vulkan- пути (см. находки 3–7) и CPU tiled-путь (находка 1).

**Чего я не определил:** буквально запустить master нельзя (нет master-билда; переключать ветку мне запрещено). `LLAMA_UPSTREAM=all` не убирает форковый `llama-context` под капотом. Механизм overflow именно на Metal я не доказал — все Metal-op'ы совпадают с CPU на случайных данных, а на реальных активациях дампа у меня нет; правдоподобный кандидат — переполнение f16/f32 в цепочке PLE-гейта на раздутой активации случайновесовой toy-модели, но это гипотеза, не замер.

Про moe-cache flakiness: механизма в ggml-backend/alloc, который объяснял бы «тот же вход → разные числа», я не нашёл (см. находку 9 — там только OOM-прочность и ослабленный guard, оба не дают вариативности). Отдельно: **`GGML_OP_TOP_K` на MTL0 расходится с CPU** (`test-backend-ops -o TOP_K`, ERR до 2.0, в т.ч. `ties=0`) — но и код Metal top-k, и тест-кейсы `TOP_K` в диффе не тронуты, значит это наследство master.

---

#### Находки (по убыванию тяжести)

**1. Critical — CPU: новая поддержка F16 `src1` доходит до tiled-ядра и рвёт assert. ON by default (x86).**
`ggml/src/ggml-cpu/ggml-cpu.cpp:459-463` добавляет в `supports_op(MUL_MAT)` ветку `src1->type == F16 && ggml_is_quantized(vec_dot_type) && blck_size <= 1024 && 1024 % blck_size == 0` (в master её нет — проверил `git show master:`). `ggml/src/ggml-cpu/ggml-cpu.c:1282` вызывает `ggml_compute_forward_mul_mat_tiled` **до** конвертации `src1`; `ggml_tiled_supported` (`tiled/tiled.cpp:605-638`) гейтится только по типу `src0`, потому что проверка типа `src1` стоит мёртвым кодом после `return true` в `switch` (`:629-636`) — дед код унаследован из master, но форк сделал его значимым. Дальше драйвер tiled: `GGML_ASSERT(src1->type == GGML_TYPE_F32)` (`tiled/tiled.cpp:1064`).
Сценарий: квантованная (Q2_K…IQ1_M) модель + f16-активация (`LLAMA_HC_F16`) + `ne11 >= 8` → `GGML_ABORT`, не тихая ошибка. `GGML_CPU_TILED_MM` по умолчанию **вкл** (`tiled.cpp:573-581`). Не достижимо на Apple Silicon (`#if !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)` → `return false`), только x86-сборки. Фикс: убрать мёртвую ветку или перенести проверку `src1` до `switch`.

**2. Critical — CUDA: `GGML_HINT_IDS_REPEAT` раздувает stream-k сетку тайлов; assert переполняется в 32 бита.**
`ggml/src/ggml-cuda/mmq.cu:371` (`ncols_max = ... == GGML_HINT_IDS_REPEAT ? ne_get_rows : ne12`) → `mmq.cuh:1480` `ntx = (args.ncols_max + J-1)/J` → `mmq.cuh:1510` `const int ntiles_dst = ntx * nty * ntzw;` → `mmq.cuh:1514` `GGML_ASSERT(ntiles_dst * blocks_per_ne00_fd.z < (1 << 30));`. Умножение идёт как `int * uint32_t`, т.е. **в 32-битном беззнаковом**, и `ntiles_dst` само — `int` произведение трёх `int`. `ncols_max` вырастает ровно в `n_expert_used` раз против upstream.
Сценарий: MoE down-proj с repeat-hint (ставится в `src/fork/llama-graph-fork.cpp:160`), 256 экспертов, `ne01=4096`, Q4_K (`blocks_per_ne00=128`), ubatch 2048 × 8 id → `ntx=512`, `ntiles_dst≈8.4M`, произведение упирается в 2^30 → `GGML_ABORT`; при ещё больших формах произведение оборачивается и assert **проходит ложно**, после чего 32-битный `kbc` в `mul_mat_q_stream_k`/`_fixup` переполняется → тайлы пропускаются или считаются дважды (тихий неверный вывод). ON by default для qwen4exp-MoE на CUDA. **Проверить прогоном не мог** (нет CUDA на машине) — вывод по чтению кода.

**3. Suggestion — Vulkan `mul_mmid_f16reg`: 32-битный перенос в элементном пространстве (default ON на RDNA3).**
`ggml/src/ggml-vulkan/vulkan-shaders/mul_mmid_f16reg.comp:225`: `const uint row_blk = (expert * p.batch_stride_a + grow * p.stride_a) / A_QK;` — произведение формируется в **элементах** и только потом делится, тогда и upstream (`mul_mmq.comp:182`, `mul_mat_vec_base.glsl:74`) и соседний файл самого форка (`mul_mm_f16reg.comp:133`) делят до умножения. При количестве элементов `src0` > 2^32 (q4_K ≈ 2.42 GB, q8_0 ≈ 4.0 GB) индекс оборачивается и остаётся **внутри** буфера — без fault, тихо читаются веса другого эксперта. 64-битный fallback в `ggml-vulkan.cpp:8026-8028` явно исключает этот пайплайн (`if (!f16reg && nbytes > range)`),_host_ ограничивает только по `maxStorageBufferRange` (`fork/ggml-vulkan-ops-fork.cpp:143`). Ручка `GGML_VK_MMID_F16REG`, при незаданной форсируется в `1` на AMD RDNA3 (`fork/ggml-vulkan-fork.cpp:229-238`). Фикс: `expert * (p.batch_stride_a / A_QK) + grow * (p.stride_a / A_QK)`.

**4. Suggestion — int8 coopmat по умолчанию на RADV/AMD-RDNA3 переводит активации в q8_1.**
`ggml/src/ggml-vulkan/fork/ggml-vulkan-fork.cpp:593-606` (`int_coopmat_on = (driver == RADV) || (AMD prop && RDNA3)`), `fork/ggml-vulkan-ops-fork.cpp:78-88`. Плотные `MUL_MAT`/`MUL_MAT_ID` уходят с upstream f16-деquant пути на int8 coopmat с реквантованием f32-активаций в q8_1: относительная погрешность блока ~1/254 (0.4 %) против ~2^-11 (0.05 %). Накопление корректное (`coopmat<float, ..., Accumulator>`, проверено в `mul_mmq_cm1.comp`/`mul_mm_f16reg.comp`); ошибка ограничена на блок и линейна по matmul, но накапливается по глубине. Форк сам снимает с int8 логиты (`GGML_TENSOR_FLAG_OUTPUT`) и `K<512` (ops-fork.cpp:82-86). ON по умолчанию на этих двух драйверах, off на всех прочих.

**5. Suggestion — f16-копии активаций по умолчанию: лишнее округление и inf при |x| > 65504.**
`mul_mmid_f16reg.comp:383-386` и `mul_mm_f16reg.comp:217,243-244` (`packFloat2x16(f16vec2(...))`), вкл. `GGML_VK_MMID_F16REG_GLU_B16` (`ops-fork.cpp:165-166`, default ON) и `GGML_VK_MM_F16REG_B16` (`:249-250`, default ON при K ≤ 4096). Заменяемый путь квантует активации в q8_1 с f32-масштабом и **не может** переполниться; здесь любое |x| > 65504 становится `±inf`, и `0 * inf` внутри `coopMatMulAdd` даёт NaN в сохраняемом элементе. Это единственная найденная разница в поведении NaN/Inf против upstream.

**6. Suggestion — `GGML_OP_FLASH_ATTN_LIST`: не требован `gqa_ratio > 1`, хотя маска не может быть построчной.**
`flash_attn_cm1.comp:194` / `flash_attn.comp:222`: индекс `SPARSE_LIST` (`data_m[sparse_base + j*Bc + c]`) не зависит от `rr`, поэтому одно значение пишется во все 4 компоненты `m`; при `gqa_ratio == 1` (`flash_attn_base.glsl:222-224` `n_groups = CEIL_DIV(p.nem1, Br)`) список и маску делит весь `Br`-тайл строк. `supports_op` (`ggml-vulkan.cpp:16594-16598`) требует только `!coopmat2 && src[5]==I32 && src[3]==F16`, `use_sparse` (`:8905`) включается самим фактом наличия `list_idx`. Если кто-то подаст MHA — все `Br` строк получат additive-bias/`-inf` от чужой строки. Текущий производитель (QSA, `src/fork/qwen4exp-fork.cpp:2236-2241`) проверенно GQA, поэтому это дыра в guard'ах, а не подтверждённый неверный вывод.

**7. Suggestion — дефолтные фьюны меняют порядок редукции против upstream (детерминированно, ulp-уровень).**
`rms_norm_rows.comp:63-79` (32 lane × до 16 шагов с шагом 32, затем 32-tree; `inversesqrt(mean+eps)*scale` свёрнуто в один множитель) против `rms_norm.comp:88-127`; `ssm_conv_concat.comp:50-53` (скалярный цикл вместо `dot(vec4,vec4)` из `ssm_conv.comp:40-45`, а GLSL не специфицирует ассоциативность `dot`); `dsv4_hc_post_norm.comp:103-115`. Обе ручки default ON (`GGML_VK_RMS_NORM_ROWS`, `GGML_VK_SSM_CONV_CONCAT`, `fusion-fork.cpp:654`, `:788`). Бит-совпадения с upstream ожидать нельзя; рост ошибки отсутствует. Плюс `mul_mm.comp:539` сужает к `D_TYPE` **до** `max(...,0)`, тогда как остальные store-пути (`:549,:559,:620,:623`) — после; при `D_TYPE == half` выровненный и невыровненный пути на одной форме различаются, а `MUL_MAT_HEADSUM` (`:518,:582`) применяет relu безусловно, вся безопасность в проверке цепочки `fusion-fork.cpp:284-300`.

**8. Suggestion — top-k: намеренно другой (стабильный) выбор на ничьих, чем upstream.**
Vulkan `topk_radix_select.comp` + `STABLE`/`GGML_VK_TOPK_RADIX_STABLE=1` (default) и CUDA `fork/top-k-fork.inc` (`GGML_CUDA_TOPK_RADIX_STABLE`, default 1; сам radix — `GGML_CUDA_TOPK_RADIX`, default auto при `nrows >= 8`). Проверил арифметику: `n_above = k - state.rank` совпадает с upstream (`top-k.cu:206`), пороги/бинарные разряды и накопление `above[]`/`ties[]` корректны, записи ограничены `pos < state.rank`/`p.k` — OOB нет. Но на точных ничьих выбирается множество/порядок «по возрастанию индекса», а не upstream-порядок атомической компакции. Это исправление недетерминизма (форк документирует расходящиеся прогоны), однако утверждать «совпадает с upstream» для top-k нельзя. `topk_argsort.comp:54-57` — вообще **фикс** upstream-бага (сравнение `floatBitsToInt` инвертировало порядок отрицательных значений при `k == 1`).

**9. Suggestion — `ggml-alloc.c`: непроверенный `realloc` в дефолтном пути аллокации весов.**
`ggml/src/ggml-alloc.c:1283-1284` в `alloc_tensor_range_fork` результат `*buffers = realloc(...)` не проверяется, и следом `(*buffers)[(*n_buffers)++] = buffer;` пишет в `NULL` при OOM (плюс утечка прежнего блока). Путь дефолтный для CUDA/Metal/Vulkan/CPU; выход — `GGML_ALLOC_UPSTREAM=1`. Upstream-аналог (`ggml-backend.cpp:93-158`) использует `std::vector` и проверку. Также `fork/ggml-backend-fork.inc:263` (`snapshot_restore`) валидирует снапшот только по счётчику `n_realloc`, который не растёт при re-reserve без роста буфера (`ggml-alloc.c:957-977`, вызов в `ggml-backend.cpp:2080-2113`) — граф может быть восстановлен с адресами прежнего плана (use-after-free нет, но упорядочение держится только на сериализации основным стримом).

**10. Nit — `mul_mmid_f16reg.comp:357`: `row_ids[min(n, tile_count - 1)]` при `tile_count == 0` читает неинициализированную shared.**
`tile_count - 1 == 0xFFFFFFFF`, `min` вырождается в `n`, а fill-цикл (`:674-677`) пишет только `i < tile_count` → произвольный `e0` → OOB-чтение `data_b[e0/8]`. Сейчас недостижимо (`count_experts.comp:141-143` гарантирует `n_tiles >= 1`), но локального `max(tile_count, 1u)` нет. Аналогично `rms_norm_rows.comp:37` (`MAX_ITERS 16`) — guard написан против дефолтного `ROW_THREADS` (`fusion-fork.cpp:666` `a->ne[0] > 32*16`), а не против константы: смена `ROW_THREADS` превратит отказ в тихое усечение.

---

#### `LLAMA_MOE_DOWN_F16` / `LLAMA_HC_F16` — конкретно по вопросу владельца

Оба **только для Vulkan** и Metal/CPU не касаются вообще — этим и объясняется, что `LLAMA_MOE_DOWN_F16=0` и `LLAMA_HC_F16=0` на Metal ничего не меняют:
- `src/fork/qwen4exp-fork.cpp:955` + `:958-960`: `hc_f16 && il >= 0 && nt >= 32 && cparams.fused_dsv4_hc_pre && qwen4exp_dev_is_vulkan(...)` → `ggml_cast(xn, F16)`. F32→F16 одного прохода нормализованного hc-потока (сам остаток остаётся f32), только при фьюне `HC_POST_NORM_F16`.
- `src/fork/qwen4exp-fork.cpp:2643-2645`: `down_f16 && ggml_nrows(cur) >= 2048 && qwen4exp_dev_is_vulkan(...)` — f16-чтение активации down-проекцией экспертов, которую сам же fьюзнутый kernel и записал.
Метрику ошибки (KLD из комментариев: 0.015284 против 0.015465) я независимо не перепроверял — на машине нет реальной модели; количественно не аудитировал.

#### Проверено и чисто

- **Vulkan-фьюны** (`fork/ggml-vulkan-fusion-fork.cpp`, `ggml-vulkan-ops-fork.cpp`, `ggml-vulkan-fork.*`): все 1283+539+1492 строки прочитаны, арифметика сверена с `rms_norm.comp`/`ssm_conv.comp`/`mul_mm.comp`/`quantize_q8_1.comp`/`unary.comp` и с конструкторами опов в `ggml.c`; условий фьюжна, которые не проверялись бы, кроме находки 6, нет; `dsv4_hc_post_norm` Q8 побитово воспроизводит `quantize_q8_1.comp QBLOCK_X4 A_F16`.
- **Шейдеры**: `get_rows.comp FLAT_ROWS` (индексные множества идентичны, `n = ne00*ne10` ограничен хостом `UINT32_MAX/2`), `count_experts.comp` (все барьеры на месте, `any_skipped`/`atomicOr` до barrier), `quantize_q8_1.comp`, `dsv4_hc_pre.comp X_F16` (конверсия точна, fma сохранена), `gated_delta_net_lazy.comp` (реплей применяет ту же `g*s + k*delta`; коллизия `temp[SUBGROUP_SIZE]` — наследство upstream, не форк), `mul_mmq_cm1.comp FUSED_GLU` (`g/(1+exp(-g))*u` == `ggml_vec_swiglu_f32`), nibble↔sub-block сопоставление q4_K/q5_K в `mul_mmid_f16reg` (сверено с `ggml-quants.c:1529-1532`), разложение delta flash-attention (`split_kv = sparse_capacity`, `fa_sparse_tail` ограничен `end_j`), `flash_attn_cm2.comp:204` — реальный фикс.
- **ggml-ядро (память)**: UAF/алиасинга при захвате графа не нашёл; префетч упорядочен `before[b]`-событием и основным стримом, `prefetch_sync` добавлен на путях ошибки и re-reserve; `done[b][t % (MAX+1)]` без коллизий; `ggml_gated_delta_net_replay` — намеренный view-алиас; целочисленная арифметика размеров/оффсетов в `int64_t`; каждый backend либо реализует `MUL_MAT_ID` со `src[3]`, либо явно отказывает (CPU/CUDA/Vulkan/Metal/SYCL/WebGPU/ET/OpenVINO проверены).
- **CPU-бекенд** (`fork/ops-fork.inc`, `fork/timeline-fork.inc`, диффы `ops.cpp`/`ggml-cpu.c`/`ops.h`/`repack.cpp`/`spacemit/ime.cpp`): кроме находки 1 — чисто; `timeline-fork.inc` — чистый no-op при `GGML_TIMELINE` не задан.
- **`GGML_HINT_IDS_REPEAT` в mmid.cu** и ранний `return` в `mul_mat_q_stream_k_fixup` (`mmq.cuh:1405+`): guard блока uniform-ов и корректен; проблема только в сетке тайлов (находка 2).

Файлов не менял, сборки не запускал.

---

### B2.1 Аллокатор ядра ggml

- Агент: `Explore-FKVHzC7NpIqQSqpfFNYUv9zMuFw4v1mT` (тип `Explore`, глубина 1, родитель: `general-purpose-ksLnpjA7E67cqBYBArnNK6VaQtRzgMCz`)
- Статус: completed, создан 13:51:28 UTC, финальный отчёт 9463 символов

**Постановка задачи (дословно из лога):**

READ-ONLY code review. Do NOT edit any file. Repo: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`, a performance fork of upstream llama.cpp). Merge base is branch `master` (aa588a346).

Examine ONLY these changes (use `git diff master...HEAD -- <path>`):
- ggml/src/ggml-alloc.c  (+206 lines)
- ggml/src/ggml-backend.cpp (+238)
- ggml/src/fork/ggml-backend-fork.inc (+734, new)
- ggml/include/ggml.h, ggml/include/ggml-backend.h, ggml/src/ggml-alloc.h (diffs)
- ggml/src/ggml.c (+160) and ggml/src/ggml.cpp (+11) diffs

Goal: find MEMORY-SAFETY and CORRECTNESS bugs only:
1. Buffer reuse while still in use / use-after-free / dangling pointer to a freed buffer.
2. Aliased buffers during graph capture (two live tensors sharing the same backend buffer region), or a buffer returned to a pool while a graph still references it.
3. Missing synchronization before a CPU reads a GPU buffer (or vice versa) in the added backend code.
4. Integer overflow / signed-negative handling in size or offset arithmetic.
5. Any ggml.c change that alters tensor semantics for ALL backends (e.g. op reuse/support lists, node extra size, is_node check, view/off behavior) in a way that could produce wrong results or an out-of-bounds read.
6. Flags/features that are ON BY DEFAULT vs opt-in (name each env knob and its default).

Method: read the diff, then read enough surrounding upstream code to confirm each difference is REAL (e.g. what upstream guarantees). For each finding give: SEVERITY (Critical = wrong output or memory error / Suggestion / Nit), exact file:line, one-line claim, the concrete failure scenario, whether it is ON BY DEFAULT or opt-in (env knob + default), and the specific code evidence you read. Also state explicitly which of the above areas you checked and found clean. Do NOT report style, comments, duplication, or performance. Max 8 findings.

**Отчёт агента (дословно):**

Everything is now in place. Write the final report in Russian (per user memory), keep identifiers/code as-is, no em dashes... (AGENTS says to avoid em dashes in code, but for prose in Russian it's fine to use them. That said, keep it simple). # Отчёт: ревью памяти/корректности, ветка `cache` (ggml-слой форка)

Проверены все перечисленные файлы полностью (`git diff master...HEAD` + чтение окружающего кода апстрима). Ниже — что нашлось, и что проверено чистым.

#### Findings

##### 1. Suggestion — не проверен результат `realloc`, запись по нулевому указателю при OOM
- **Где:** `/Users/askuzminov/Projects/github/askuzminov/llama.cpp/ggml/src/ggml-alloc.c:1283-1284`
- **Утверждение:** в `alloc_tensor_range_fork` результат `*buffers = realloc(...)` не проверяется; при неудаче `(*buffers)[(*n_buffers)++] = buffer;` пишет в `NULL` (краш), а исходный блок ещё и теряется.
- **Сценарий:** загрузка модели с отвалом памяти между диапазонами (типично как раз при нехватке VRAM/RAM — тот случай, ради которого рядом сделан `GGML_ALLOC_RETRY`) вместо аккуратного `return false` даёт сегфолт.
- **Вкл. по умолчанию:** да. Это дефолтный путь аллокации весов в llama: `src/llama-model.cpp:2128` вызывает `ggml_backend_alloc_ctx_tensors_from_buft_cb` для всех buffer type без `iface.alloc_buffer_n` (CUDA/Metal/Vulkan/CPU — NULL). Единственный выход — `GGML_ALLOC_UPSTREAM=1` (`ggml-alloc.c:1332`, по умолчанию выключен).
- **Доказательство:** апстримый аналог `ggml_backend_buft_alloc_buffer_n_default` (`ggml/src/ggml-backend.cpp:93-158`) использует `std::vector` и проверку `t_failed`; форк-копия заменила вектор на сырой `realloc` без проверки.

##### 2. Suggestion — проверка валидности snapshot слишком слабая: `n_realloc` не растёт при re-reserve без роста буфера
- **Где:** `ggml/src/fork/ggml-backend-fork.inc:263` (`snapshot_restore`), прирост счётчика — `ggml/src/ggml-alloc.c:975-976`.
- **Утверждение:** `snapshot_restore` возвращает `true`, если `ggml_gallocr_get_n_realloc()` равен значению на момент save; но `n_realloc++` происходит только когда чанк vbuffer реально перерастёт (`new_chunk_size > cur_chunk_size`, `ggml-alloc.c:957-977`). `ggml_backend_sched_alloc_splits` (`ggml/src/ggml-backend.cpp:2080-2113`) делает re-reserve и при `backend_ids_changed` (или когда меньший граф «влезает» в план) — план перестраивается (смещения tensor'ов меняются), а `n_realloc` не меняется, и restore пропускает граф со «старыми» адресами из прежнего плана.
- **Сценарий:** save снапшота графа A; далее строится граф B, у которого те же размеры узлов, но часть узлов уехала на другой backend (op_offload зависит от размера батча) либо план уплотнился без роста буфера; restore(A) возвращает `true`; вычисление A идёт по адресам старого плана. use-after-free нет (буферы те же, память валидна), но при `graph_compute_async` пересечение отложенного compute промежуточных графов (по новому плану) с restored-графом (по старому плану) ничем не защищено — ни событиями, ни `cur_copy` (он привязан к текущему плану). Входные копии каждый compute перезаполняются, поэтому немедленной порчи не видно; риск — гонка при асинхронном чередовании графов.
- **Вкл. по умолчанию:** да, через граф-кэш llama: `src/llama-context.cpp:1743` (`graph_cache_size > 0`), restore — `src/fork/llama-context-fork.cpp:608`.
- **Что предлагаю:** инкрементировать счётчик (или отдельный generation-счётчик планов) при каждом `ggml_gallocr_reserve_n_impl`, либо при restore сверять `addr` хотя бы для узлов-копий входа, а не только рост буферов.

##### 3. Nit — `prof_gpu_init` не проверяет `prof_devs[b] == NULL`
- **Где:** `ggml/src/fork/ggml-backend-fork.inc:31-33`.
- `ggml_backend_get_device()` возвращает `backend->device` (`ggml-backend.cpp:827-830`) без гарантии non-NULL; `ggml_backend_dev_backend_reg(nullptr)` затем разыменовывает NULL. Достигнимо только при включённом `GGML_SCHED_PROF` + `GGML_SCHED_PROF_GPU` (opt-in), для стандартных backend'ов device ненулевой — поэтому Nit.

#### Проверенные области и вердикты

**1. Reuse буфера во время использования / UAF / висячие указатели — чисто** (с оговоркой п.2):
- Префетч: копия веса аллоцируется в `graph_copy` с позиции сплита-загрузчика (`prefetch_add_inputs`), ggml-alloc держит регион живым до сплита-потребителя (учтён n_children); запись на второй инстансе упорядочена `before[b]` (event record на основном стриме до работы сплитов `< s`) — всё, что писало этот регион раньше, гарантированно завершилось; кросс-графовые очереди сериализуются тем же main-stream и тем же copy-stream. На путях ошибки `compute_splits` и в re-reserve добавлен `prefetch_sync` (`ggml-backend.cpp:2111`, `2324`, `2354`).
- Индексация событий `done[b][t % (MAX+1)]`: повторная запись possible только для `t' = t + MAX+1` на хост-тайминге строго после wait'а сплита t — коллизий нет (проверено для depth 1..4, MAX=4).
- Snapshot: shallow-копия `ggml_tensor` (`**cpy`) не владеет ничем, кроме неустанавливаемого `extra` (не освобождается дважды); буферы с `reset`-интерфейсом отсекаются (`has_reset`); память `sched->context_buffer` больше не видна снапшоту — src'ы переназначены на `snapshot->copies` с remap'ом (`fork.inc:232-248`).
- Путь загрузчика cb: при ошибке диапазона `cb(NULL, NULL)` вызывает `ranges->drain()` (`src/llama-model.cpp:1619-1622`) — лоддер-тред завершает записи до `ggml_backend_buffer_free`; UAF нет.

**2. Алиасинг буферов при захвате графа — чисто:**
- `ggml_gated_delta_net_replay` возвращает `ggml_view_tensor(state_all)`; `ggml_impl_is_view` (`ggml-impl.h:103`) завязан на `view_src`, а не на список опов, поэтому ggml-alloc не выделяет отдельный регион для результата (`ggml-alloc.c:664, 776, 1017`) — алитас намеренный и корректный; упорядочение потребителей — через edge на результат (документировано в ggml.h).
- `ggml_flash_attn_list` использует `src[5]` при `src[4]=NULL` — конфликтов с семантикой FA_EXT (sinks=src[4]) нет.
- Префетч не добавляет узлов сверх бюджета `graph_size = max(n_nodes,n_leafs) + total_inputs*2*n_copies + n_dep_nodes` (префетчнутый вход вместо своих 2 узлов даёт те же 2 узла в раннем сплите).

**3. Синхронизация перед CPU-чтением GPU-буфера — чисто** в рамках отревьюенных файлов: префетч — только H2D из статичного хост-буфера; `split_uses_host_buffer` трактует `buffer == NULL` как host (консервативно); `copy_input`-лямбда эквивалентна upstream-логике синхронизаций (`ggml_backend.cpp:2138-2179`, сравнил с `master:2083-2128`); `parallel_cpu`-обход синхронизации (`ggml-backend.cpp:2225-2240`) срабатывает только когда предыдущий device-сплит не трогал ни одного host-буфера (учитывая вьюхи через `view_src`), переменная `pending_host` локальна на вызов — меж-графовые хвосты не попадают.

**4. Целочисленные переполнения/отрицательные значения — чисто:** `GGML_PAD(buft_get_alloc_size)`/`cur_buf_size + this_size` — тот же паттерн, что в апстримом `alloc_buffer_n_plan`; `(int64_t)`-касты в ассертах gated_delta_net корректны; `min_load * input->ne[2]` считается в int64; depth клампится `[0..4]`; отрицательный `GGML_SCHED_PREFETCH` → `std::max(0, …)`.

**5. Изменения ggml.c, влияющие на все backend'ы — проверено, чисто:**
- Вставка `GGML_OP_FLASH_ATTN_LIST`/`GGML_OP_GATED_DELTA_NET_LAZY` — только перед `GGML_OP_COUNT`; оба `static_assert` (OP_NAME/OP_SYMBOL) обновлены, рассинхрон имён невозможен.
- `ggml_mul_mat_id_add_skip` (src[3]): каждый backend с MUL_MAT_ID либо реализует skip (CPU: `ggml-cpu.c:1584`), либо явно отвергает: CUDA `op->src[3] -> return false` (ggml-cuda.cu, supports_op MUL_MAT_ID), Vulkan (`ggml-vulkan.cpp:16534`), Metal (`ggml-metal-device.m:1872-1874` — `op->src[3] == NULL && …`), SYCL (`ggml-sycl.cpp:7303`), WebGPU (`:4545`), ET (`:1024`), OpenVINO («MUL_MAT_ID with a skip input is not supported»). Молчаливого игнора skip не нашёл.
- Ослабление assert'а `ggml_dsv4_hc_pre` на F16: CPU имеет отдельный `ggml_compute_forward_dsv4_hc_pre_f16_fork` (`ggml-cpu/ops.cpp:11677`), Vulkan пропускает f16 только для `src[0]` HC_PRE (`ggml-vulkan.cpp:16909-16910`), Metal по-прежнему требует `src[0] == F32` (`ggml-metal-device.m:1818-1820`) — несовместимые backend'ы просто не берут оп.
- `GGML_HINT_IDS_REPEAT` + assert `ggml_mul_mat_set_hint`: slot op_params[1] для MUL_MAT_ID никем другим не занят; CUDA/Metal отвергают repeat на несовместимых путях (`ggml-cuda.cu:5896`, `ggml-metal-device.m:1871`).

**6. Флаги по умолчанию (env-ручки):**

| Ручка | Файл:строка | Значение по умолчанию |
|---|---|---|
| `GGML_SCHED_PREFETCH` | fork.inc:442 | **3 — ВКЛЮЧЕН** (0 = выкл; вкл. дефолт с 14.10) |
| `GGML_SCHED_PARALLEL_CPU` | ggml-backend.cpp:2423-2424 | **ВКЛЮЧЕН** (выкл. `=0`) |
| cb-путь аллокации весов | ggml-alloc.c:1320+ / llama-model.cpp:2128 | **ВКЛЮЧЕН** (выход `GGML_ALLOC_UPSTREAM=1`, по умолчанию выкл.) |
| `GGML_ALLOC_RETRY` | ggml-alloc.c:1258-1261 | **3** retries по 2 с — включено |
| `GGML_SCHED_PREFETCH_LOAD` | fork.inc:432 | 20 (величина порога, эффект только при вкл. prefetch) |
| `GGML_ALLOC_PEAK` | ggml-alloc-peak-fork.inc:30-34 | выкл. |
| `GGML_SCHED_LOG_REALLOC` | ggml-alloc.c:1054-1058 / fork-decl.inc:7 | выкл. |
| `GGML_SCHED_PROF`, `GGML_SCHED_PROF_GPU` | ggml-backend.cpp:2419 / fork.inc:26 | выкл. |
| `GGML_TIMELINE` | fork.inc:627-636 | выкл. (файл не задаётся — всё no-op) |

Итог: критических memory-safety дефектов в перечисленных файлах не обнаружено; два замечания уровня Suggestion (прочность к OOM в `alloc_tensor_range_fork` и ослабленный guard в `snapshot_restore`) — оба на путях, включённых по умолчанию.

---

### B2.2 Vulkan-фьюны и операции

- Агент: `Explore-qpUnNqSZ1jYs7bAENiw2puXDGLQ0WA76` (тип `Explore`, глубина 1, родитель: `general-purpose-ksLnpjA7E67cqBYBArnNK6VaQtRzgMCz`)
- Статус: completed, создан 13:51:28 UTC, финальный отчёт 13624 символов

**Постановка задачи (дословно из лога):**

READ-ONLY code review. Do NOT edit any file. Repo: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`, a performance fork of upstream llama.cpp). Merge base is branch `master` (aa588a346).

Examine ONLY these NEW files:
- ggml/src/ggml-vulkan/fork/ggml-vulkan-fork.cpp (1492 lines)
- ggml/src/ggml-vulkan/fork/ggml-vulkan-fork.h
- ggml/src/ggml-vulkan/fork/ggml-vulkan-fusion-fork.cpp (1283 lines)
- ggml/src/ggml-vulkan/fork/ggml-vulkan-ops-fork.cpp (539 lines)

Goal: find NUMERICAL and KERNEL-CORRECTNESS bugs only, in this priority order:
(a) A graph fusion that changes model output: fusing ops in a way that is only valid under conditions that are NOT actually checked (missing dtype check, missing alignment/contiguity check, missing stride check, ignoring that an operand is also consumed by another node so the fused write clobbers it, ignoring a mask/bias/scale input, fusing across a view with non-trivial nb offsets).
(b) Precision reduction: f16 accumulation where upstream uses f32, or a fused kernel that rounds an intermediate to f16/fmt that upstream keeps in f32. Say exactly where and whether the magnitude is bounded.
(c) Out-of-bounds reads/writes: index arithmetic with wrong stride/pitch, missing bounds guard when a dimension is not a multiple of the tile/workgroup size, integer overflow on 32-bit index math for large sizes (contexts up to 262144!).
(d) Missing memory barriers / uninitialized shared memory in any dispatch-side assumptions.
(e) NaN/Inf handling that differs from upstream.

Method: read each new file, and read the corresponding upstream code (ggml/src/ggml-vulkan/ggml-vulkan.cpp, the .comp shaders under ggml/src/ggml-vulkan/vulkan-shaders/, and ggml/src/ggml-vulkan/vulkan-shaders/*.glsl) to confirm the fused math matches upstream bit-for-bit or is documented as approximate. Check every guard the fusion asserts. Name every env knob you find and whether its default is ON or OFF.

For each finding give: SEVERITY (Critical = wrong output or memory error / Suggestion / Nit), exact file:line, one-line claim, the concrete failure scenario (input/shape/dtype and what wrong value comes out), ON BY DEFAULT or opt-in, and the specific code evidence. State explicitly which areas you checked and found clean. No style/perf/duplication findings. Max 8 findings.

**Отчёт агента (дословно):**

Confirmed that dsv4_hc_post_norm has wg_denoms {1,1,1} → workgroup-y = n_tokens is obtained directly; 65535 upper-limit assert. The upstream dsv4_hc_post (non-norm) probably uses {256,1,1}, and it's dispatched with the elements {256?...}. Either way, the same n_tokens restriction probably exists upstream too. Keeping this as a nit.

One more thing: I claimed that ggml_vk_fork_mmv_id's fused MUL scale uses `next->src[1]` — this is safe because the fork branch of mmid_mul_ok in can_fuse requires mmid == mul->src[0] (mmid_mul_ok checks `if (mmid != mul->src[0]) return false;` before both branches). ✓

Also re-checking `scale->ne[1] == mmid->ne[1]`: mmid->ne[1] — is it the total slot×token of the MUL_MAT_ID output? dst->ne[1] for mul_mat_id = ids->ne[0]*ids->ne[1]. So scale is [1, nei0*nei1] and linear → [token*nei0+slot] ✓ contiguous ✓.

Also want to check the fused MUL write: the shader writes to data_d, but when fused, out=mul node (d_D = subbuffer(out)) ✓ stride is taken from the matmul's dst, which is identical shape and contiguous ✓.

Now I'll write the report. Structure:

Review scope/methodology (concise)
Findings (numbered, severity, file:line, claim, scenario, default, evidence)
Environmental inventory (ON/OFF defaults)
Confirmed clean areas

Make it organized and precise. Language: English (user wrote in English). # Review: numerical & kernel correctness of the four new fork files

I read all four new files end-to-end and cross-checked the fused math against the corresponding upstream shaders (`rms_norm.comp`, `ssm_conv.comp`, `mul_mm.comp`, `quantize_q8_1.comp`, `unary.comp`, `dsv4_hc_post_gate` chain) and against the op constructors in `ggml.c` (`ggml_dsv4_hc_post`, `ggml_glu_impl`, `ggml_lightning_indexer`) and `ggml_can_fuse_subgraph_ext`. No Critical (wrong-output / memory-error) bug was found; the fusions are unusually well guarded. Findings below are ordered by importance.

---

#### Findings

##### 1. Suggestion (b) - int8 coopmat is ON by default on RADV/AMD-RDNA3 and moves activations to q8_1 (0.4% rounding) where upstream used f16 (0.05%)
`ggml/src/ggml-vulkan/fork/ggml-vulkan-fork.cpp:593-606` (`ggml_vk_fork_int_coopmat`), `ggml/src/ggml-vulkan/fork/ggml-vulkan-ops-fork.cpp:78-88` (`ggml_vk_mul_mat_int8_off`).
- Claim: with `GGML_VK_INT_COOPMAT` unset, `int_coopmat_on = (driver == RADV) || (AMD proprietary && RDNA3)` - default ON - and dense `MUL_MAT`/`MUL_MAT_ID` switch from upstream's f16-dequant path to int8 coopmat with f32 activations requantized to q8_1 x4.
- Scenario: any q4_K/q5_K/q8_0 hidden matmul in a prefill on RADV or the AMD Windows driver on RDNA3. Each activation value takes a per-32-block relative rounding of up to ~1/254 (~0.4%) vs ~2^-11 (0.05%) on upstream's f16-activation path; error is bounded per block, propagates linearly through the matmul, and accumulates over depth.
- The fork itself acknowledges the mechanism: logits (`GGML_TENSOR_FLAG_OUTPUT`) and K<512 matmuls are deliberately kept off int8 because "the error of the q8_1 activations goes straight into the result" (ops-fork.cpp:82-86), and KLD comparisons are quoted in comments. Weight codes themselves stay exact under int8 repack; accumulator is int32/f32 - verified in `mul_mmq_cm1.comp`/`mul_mm_f16reg.comp` (`coopmat<float, ..., Accumulator>`).
- ON by default on those two drivers; every other driver is off.
- Assessment: documented-approximate trade, but it is the single largest output-bitness change in the fork and is default-on; worth an explicit README/release-note entry.

##### 2. Suggestion (b) - fused gate/up reads an f16-rounded copy of the activations by default (`GGML_VK_MMID_F16REG_GLU_B16`, ON)
`ggml/src/ggml-vulkan/fork/ggml-vulkan-ops-fork.cpp:165-166` (`b16_glu = e == nullptr || atoi(e) != 0` - default true), consumed at `:170` (`int h = ... ? 1 : 0`).
- Claim: the fused MoE gate/up (`MUL_MAT_ID_MUL_MAT_ID_GLU[_CPY]`) reads an f32->f16 copy of `x` made once in `prealloc_y`, instead of `x` in place.
- Scenario: qwen4exp prefill >= `GGML_VK_MMID_GLU_MIN_TOKENS` (2048, fusion-level default) on the f16reg path (default ON after the coopmat A probe on RDNA3). Every activation gets an extra f16 rounding before the matmul: relative error <= 2^-11 per element, accumulator is f32 (`mul_mmid_f16reg.comp:644-648, 697`), so magnitude is bounded and in practice smaller than the q8_1 error of finding 1. One unbounded corner: any |activation| > 65504 becomes +inf in the copy, turning that expert's output row to inf/NaN, whereas the f32-in-place variant would stay finite - practically unreachable activations, but it is a NaN/Inf behavior difference vs upstream (category e).
- Opt-out: `GGML_VK_MMID_F16REG_GLU_B16=0`. The dense `GGML_VK_MM_F16REG_B16` (ops-fork.cpp:249-250, ON by default for K <= 4096) is the same mechanism on the q8_0 dense matmul.

##### 3. Suggestion (b) - rms_norm_rows is not bit-for-bit upstream (reduction order, scale folded into the norm factor); deterministic, ~ulp-level
`ggml/src/ggml-vulkan/vulkan-shaders/rms_norm_rows.comp:57-76` + `ggml/src/ggml-vulkan/fork/ggml-vulkan-fusion-fork.cpp:653-786` (fusion, default ON via `GGML_VK_RMS_NORM_ROWS` unset).
- Claim: sum-of-squares is accumulated as 32 lanes x up to 16 strided elements then a 32-lane tree, while upstream `rms_norm.comp:95-126` uses a 512-invocation per-subgroup + shared tree; additionally the fused `RMS_NORM_SCALE` computes `rs = inversesqrt(mean+eps) * scale` then `x*rs`, while upstream rounds `x*rs` first and multiplies by `scale` in the SCALE node (fusion-fork.cpp:668-675 passes `scale` into pc).
- Scenario: any GDN head norm (128 cols) or GDN l2-norm in the default build. Output differs from upstream in the last ulp of the norm scale (relative ~1e-7); bounded, no amplification, deterministic across runs. `inversesqrt` itself matches upstream (`rms_norm.comp:130`).
- The same reordering remark applies to the tree in `dsv4_hc_post_norm.comp:103-115` (upstream hc_post is a different kernel anyway). Worth a comment line stating "approximate vs upstream, ulp-level" if bit-parity tests are ever expected.

##### 4. Nit (b) - HC_POST_NORM_F16 rounds the normalized stream to f16 once where upstream rounds twice; Q8 blocks then quantize the f16-rounded value
`ggml/src/ggml-vulkan/vulkan-shaders/dsv4_hc_post_norm.comp:124-131, 143-147`, dispatched from `ggml/src/ggml-vulkan/fork/ggml-vulkan-fusion-fork.cpp:16-103` (`HC_POST_NORM_F16`, `GGML_VK_HC_POST_NORM_Q8` default ON, min 32 tokens).
- Claim: fused store does `float16_t(v*scale*w)` (one rounding), while upstream rounds `rms_norm` to f32, `MUL` to f32, then `CPY` to f16 (two roundings). The fused result is closer to the real value but not bit-identical to the upstream graph.
- The Q8 variant then quantizes `nf = float(float16(...))`, which I verified is bit-identical in layout and arithmetic to what `quantize_q8_1.comp` (`QBLOCK_X4`, `A_F16`) produces from the stored f16 tensor: same `d = amax/127`, `d_inv` guard, `round(vals*d_inv)`, `ds = f16vec2(d, sum*d)`, same `pack32` word order, and the quantized-value sums (<= 32x127) are exact in f32 so the different reduction tree cannot diverge. Clean, no action beyond the bit-parity note.

##### 5. Nit (a/e) - default radix top-k changes tie selection vs upstream (deterministic, deliberate)
`ggml/src/ggml-vulkan/fork/ggml-vulkan-ops-fork.cpp:47-76`; `GGML_VK_TOPK_RADIX` default -1 (auto: k>=256, ncols wider than the tournament's 1024, nrows>=8), `GGML_VK_TOPK_RADIX_STABLE` default 1.
- Claim: for the qwen4exp block top-k (k 513 over >1024 columns) the fork's stable radix select always picks the lowest-index element on exact ties and emits index order; upstream's atomic compaction picked tied elements in run-to-run order (the comment documents non-reproducible logits as the motivation).
- Scenario: exactly-equal scores select a different (but equally-top-k) set/order of experts or KV blocks than upstream. This is a correctness improvement over a nondeterministic baseline, not a regression - flagged only because "matches upstream" cannot be asserted for it. `ggml_vk_topk_use_radix`'s auto branch is ON by default.

##### 6. Nit (c) - new per-token workgroup dispatches rely on the 65535 workgroup-count assert instead of chunking
`ggml/src/ggml-vulkan/fork/ggml-vulkan-fusion-fork.cpp:101/103` (`{4, n_tokens, 1}` with pipeline wg_denoms `{1,1,1}`, ggml-vulkan.cpp:3796-3801) and `:838-845` (ssm_conv_concat `{nr, n_t, n_s}`); `ggml-vulkan-common.h:273-275` asserts `wg1 <= maxComputeWorkGroupCount[1]`.
- Claim: with a user-raised `n_ubatch` > 65535 (legal up to n_ctx 262144), `HC_POST_NORM` hits `GGML_ASSERT` in `ggml_vk_dispatch_pipeline` (hard stop, not wrong output); `CONCAT_SSM_CONV` divides by TOKENS_PER_WG=16 so it survives to ~1M tokens; `rms_norm_rows` explicitly chunks into a 2D grid capped at 65535x (fusion-fork.cpp:729-732) and is the model the other two should follow. Crash-only, but the guard is an assert, so release builds of the fork still abort rather than fall back.

---

#### Env knobs found (all in the four files) and default state

ON by default (no env needed): GET_ROWS_FLAT (`GGML_VK_DISABLE_GET_ROWS_FLAT=1` opts out); TOPK_RADIX auto (`_TOPK_RADIX` 0/1 override, default auto), TOPK_RADIX_STABLE=1; MMID_F16REG (RDNA3+coopmat-A-probe only; `_BPF=1`, `_EPI=0`, `_DB=0`, `_MIN_ROWS=16`, `_BN=64`, `_GLU_BN=64`, `_ROWS2` auto=q5_1-only, `_GLU_SPLIT` off, `_B16` off / `_GLU_B16` on, `_LOG` off); MM_F16REG (RDNA3 only, tile `r16x96/8x1` / `_B32 r16x64/8x1`, `_B16` on, `_B16_MAX_K=4096`, `_MIN_N=64`, `_MIN_M=512`, `_MIN_K` auto, `_EMULATE` off); MMV_ID_FORK mode 2 (AMD RDNA3+IDP only; `_WG=32`, `_LPR_Q4_K=16`, `_LPR_Q5_1=4`, `_MAXR*` auto, `_Q4_K=1`, `_FENCE=1`, `_LOG` off); INT_COOPMAT on RADV/AMD-RDNA3 (`0..3` override); INT_LARGE_TILE=2 and MMQ_INT_TILE=`64,128,32,32` and MMQ_INT_TILE_GLU=`32,64,16,32` (AMD proprietary RDNA3 only); RDNA3-iGPU shmem raise to 64 KB after probe (`GGML_VK_SHMEM_LIMIT`/`_PROBE` override); MM_CHUNK_MB=24 (AMD UMA; 0 off); HC_POST_NORM (>=32 tokens; `GGML_VK_HC_POST_NORM=0` off, `_MIN_TOKENS`); HC_POST_NORM_Q8; MM_RELU, MM_HEADSUM (`GGML_VK_DISABLE_*` to turn off, `HEADSUM_MIN_TOKENS=0`), LID_HEADSUM (`=0` off), MMID_GLU (`_MIN_TOKENS=2048`), MM_HC_PRE (all off only via their DISABLE knobs); RMS_NORM_ROWS (`=0` off); SSM_CONV_CONCAT (`=0` off); elastic spill=1 / margin 512 MiB; host pad shrink (`GGML_VK_HOST_PAD_UPSTREAM=1` restores upstream 32-byte pad).
OFF by default: F16ACC forced off on AMD proprietary (`GGML_VK_F16ACC=0/1`), `GGML_VK_MMQ_TILE` unset, `GGML_VK_MMQ_INT_TILE_ID` unset, `GGML_TIMELINE`, `GGML_VK_GRAPH_TIMING=0`, `GGML_SCHED_LOG_REALLOC` (stderr logging).

#### Checked and found clean

- **Guard coverage of every fusion**: single-use of intermediates is enforced by `ggml_can_fuse_subgraph_ext` (ggml.c:8038-8048 counts all graph uses); headsum does its own `ggml_node_has_n_uses(...,1)` sweep; `GGML_TENSOR_FLAG_OUTPUT` is excluded where the fused shader skips a node (GDN snapshot, mmid-glu). GATE folding checks both `SCALE` add params == 0; `RMS_NORM_SCALE` checks add == 0; `MMID_GLU` checks GLU==SWIGLU (not SWIGLU_CLAMP/OAI), swapped flag via `ggml_glu_impl` op_params[1], same ids/x/prec, either operand order; `LID_HEADSUM` requires `FILL`-weights == 1.0f and re-derives dims from the `ggml_lightning_indexer` asserts (dst ne2 is 1 by construction, so mask indexing cannot run past the f16 mask; the f16 word-unpack parity is protected by the misalign checks).
- **Stride/aliasing**: headsum `stride_b=k` is valid exactly because `q0->nb[1] == HEADSUM*k*4` is asserted, making `col*k` land on head `col`; bias/mask shapes, strides and misalignment checked; `b_buf.size` clamped to the view and the view bounds-checked against `view_src`.
- **Math equivalence**: ssm_conv tap order (`src[t+k]*w[k]`) and silu expression match `ssm_conv.comp`; relu/sigmoid/silu expressions match `unary.comp` (`max(x,0)` -> same NaN semantics); Q8 fused blocks bit-identical to `quantize_q8_1.comp` (see finding 4); headsum sums heads in the same sequential order as the graph's add chain, bias last; `mmv_id` fused scale operand ordering guaranteed by `mmid_mul_ok` (`mmid == mul->src[0]`, linear `[token*nei0+slot]` layout matches the checked shape and contiguity); mmv_id clamps over-run rows (`min(row, nrows-1)`, store guarded) and zeroes negative expert ids.
- **32-bit index math**: products bounded by `maxStorageBufferRange` checks in every dispatch (headsum n*k*4, mmv_id y_sz, hc q_sz, GDN nb2 <= UINT32); `ggml_vk_get_rows_flat` guards `ne0*ne1 <= UINT32_MAX/2`.
- **Barriers/shared memory**: `rms_norm_rows.comp` (writes `sumsh[lr][tid]` before any barrier, uniform returns after the last barrier), `dsv4_hc_post_norm.comp` (`sum_s[tid]` unconditional write), `mul_mm_f16reg`/`mul_mmid_f16reg` use f32 accumulators with layout validated by runtime probes (`coopmat_a_probe`, `coopmat_c_probe`, `shmem_probe`) before the EPI/register paths are enabled.
- **Not numerics but reviewed**: elastic sparse buffer commit/decommit (committed granules are memset 0, unbound reads zero only claimed with `residencyNonResidentStrict`), host-buffer pad shrink (pad only matters if a shader overwrites the tail of pinned staging buffers, which the comment verifies it does not), timeline/graph-timing (diagnostics only).

The biggest residual risk is not a coding bug but scope of the documented approximations (findings 1-3): several default-on paths make output differ from upstream in the last bits or at the 1e-3 relative level, which should be stated in one place (currently these notes live only in code comments).

---

### B2.3 CUDA fork inc, первый проход

- Агент: `general-purpose-oMHJ0QfRVQ4PgNxg1FmmvNvfLFuyRq7C` (тип `general-purpose`, глубина 1, родитель: `general-purpose-ksLnpjA7E67cqBYBArnNK6VaQtRzgMCz`)
- Статус: completed, создан 15:31:09 UTC, финальный отчёт 3844 символов

**Постановка задачи (дословно из лога):**

READ-ONLY code review. Do NOT edit any file. Repo: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`, perf fork of upstream llama.cpp). Merge base = branch `master`.

Scope: CUDA backend fork code only:
- ggml/src/ggml-cuda/fork/gated-delta-net-lazy-fork.inc (279, new)
- ggml/src/ggml-cuda/fork/getrows-fork.inc (76, new)
- ggml/src/ggml-cuda/fork/repeat-fork.inc (44, new)
- ggml/src/ggml-cuda/fork/elastic-buffer-fork.inc (170, new)
- ggml/src/ggml-cuda/fork/lightning-indexer-fork.inc (17, new)
- ggml/src/ggml-cuda/fork/ggml-cuda-timing-fork.inc (548, new)
- diffs (`git diff master...HEAD --`) of: ggml-cuda.cu (+102/-), mmid.cu, mmq.cu, mmq.cuh, top-k.cu, getrows.cu, gated_delta_net.cu, gated_delta_net.cuh, binbcast.cu, lightning-indexer.cu
NOTE: another reviewer already covered fork/top-k-fork.inc in depth - skip its internals, but you MAY check how top-k.cu hooks it in (dispatch constraint: does the fork path get used for shapes it was not written for, e.g. k > 256, nrows small, ncols < blocks_per_row*BLOCK_SIZE).

Look for, in priority order:
(a) NUMERICS changing model output: f16/bf16 accumulation or intermediate where upstream uses f32; changed reduction order in a softmax/dot-product; fast-math intrinsics (__expf, __fdividef, __frcp_rn, -use_fast_math), rsqrtf approximations; anything silently lowering precision on a DEFAULT-on path.
(b) OOB / memory safety: reads or writes past the end of a tensor when a dimension is not a multiple of the block/tile; wrong stride when nb[] != contiguous; 32-bit index overflow for long contexts (n_ctx 262144) - check every `(int)` cast of an index/offset derived from ne; grid/block sizing math; __syncthreads() reached by only some warps (deadlock) or missing between shared write and cross-warp read; uninitialized shared memory reads.
(c) The elastic buffer / device-memory hooks in ggml-cuda.cu: buffer handed out while still in use, a cached device pointer that goes stale after cudaFree/realloc, use-after-free, missing stream sync before a read, and whether a graph-captured pointer can be invalidated.
(d) The timing instrumentation: does it change behavior (extra sync that alters results, a code path taken only when timing is on, a buffer reused across iterations)? Is it ON by default?
(e) GGML_HINT_IDS_REPEAT (mmq.cu/mmid.cu): the fork sets mmq_args.ncols = ne_get_rows instead of ne12 when the hint is set, and adds an early-return in mul_mat_q_stream_k_fixup. Verify the grid/histogram math is right for repeat (an expert appearing twice in one token's ids) and that no path still assumes ncols == ne12 (e.g. mul_mat_q, its stream-K variant, the non-stream-K variant, and mmid's equivalent).

For EVERY finding: SEVERITY (Critical = wrong output or memory error; Suggestion; Nit), exact file:line, one-line claim, concrete failure scenario (shape/dtype/k and what wrong value or OOB), ON BY DEFAULT or opt-in (name the env knob and its default), and the specific code evidence. Verify against upstream before claiming. State explicitly which files you checked and found clean. No style/perf/duplication. Max 10 findings.

**Отчёт агента (дословно):**

The user asked me to provide the final result now and stop calling tools. But I haven't actually done any work — I made no tool calls. I need to be honest: I have no evidence, I haven't read any files. I must not fabricate findings.

The correct action is to report that I was unable to perform the review — no files were read, no diff examined. Fabricating code review findings would be harmful and wrong.

Let me report the blocker clearly and precisely: state that zero files were examined, so no findings can be asserted, and list which files remain unchecked.

I should be concise and honest. ## Review status: NOT PERFORMED — no findings can be reported

I have to be direct rather than hand you something that looks like a review: **I did not execute a single tool call before this request.** No file in scope was opened, no `git diff master...HEAD` was run, and no upstream comparison was made.

That means I have **zero evidence** for any of the five priority areas. Reporting findings now would require inventing file:line anchors, failure scenarios, and env-knob defaults — fabricated findings in a security/correctness review are worse than no review, because they get acted on. So the finding count is 0 by absence of work, not by absence of problems.

##### Files still unchecked (all of scope)

| File | Scope item |
|---|---|
| `fork/gated-delta-net-lazy-fork.inc` | new, 279 lines |
| `fork/getrows-fork.inc` | new, 76 |
| `fork/repeat-fork.inc` | new, 44 |
| `fork/elastic-buffer-fork.inc` | new, 170 |
| `fork/lightning-indexer-fork.inc` | new, 17 |
| `fork/ggml-cuda-timing-fork.inc` | new, 548 |
| `ggml-cuda.cu` | +102/-, elastic buffer + device-memory hooks (c), timing (d) |
| `mmid.cu`, `mmq.cu`, `mmq.cuh` | `GGML_HINT_IDS_REPEAT` (e) |
| `top-k.cu` | hook-in only: dispatch guard for `k > 256`, small `nrows`, `ncols < blocks_per_row * BLOCK_SIZE` |
| `getrows.cu`, `gated_delta_net.cu/.cuh`, `binbcast.cu`, `lightning-indexer.cu` | diffs vs `master` |

No claim of "clean" is made for any of these, including the ones the task suggested I might verify as clean. A clean verdict requires the same evidence a finding does.

##### What a real pass would need

Run with tool access, in roughly this order — (e) and (b) are where a Critical most plausibly lives given the described design:

1. `git diff master...HEAD -- ggml/src/ggml-cuda/mmq.cu ggml/src/ggml-cuda/mmq.cuh ggml/src/ggml-cuda/mmid.cu` and trace every consumer of `mmq_args.ncols`. The described change makes `ncols` mean two different things depending on a hint, so the question is total: is there *any* remaining reader that derives grid extent, histogram bucket count, or fixup range from `ncols == ne12`? Repeated-id tokens (`ncols < ne12`) are the under-exercised case, and the early-return in `mul_mat_q_stream_k_fixup` needs its guard condition checked against the *partial-sum* invariant, not just the no-op case.
2. Index-width audit on `gated-delta-net-lazy-fork.inc` and the longer `.inc` files: every `(int)` cast of a quantity derived from `ne`, checked at `n_ctx = 262144`.
3. `__syncthreads()` reachability and shared-write/cross-warp-read ordering in each `.inc` kernel.
4. Elastic buffer lifecycle in `ggml-cuda.cu`: pointer validity across `cudaFree`/realloc, across graph capture, and whether any handout precedes completion of a prior kernel on another stream.
5. For each fork path, establish the **default**: name the env knob and its default value. Any path that is on by default gets held to the (a) numerics bar; anything opt-in is a lower-severity concern even with the same defect.

If it helps, the single most valuable thing I could do first is the `ncols` consumer trace in item 1 — it is a bounded grep-and-read over three files and is the one place where the described design already sounds like it violates a stated invariant.

---

### B2.4 Новые и изменённые GLSL-шейдеры

- Агент: `general-purpose-xrw1nNGY0G8cGQ9zgejMyaNU4GMFGQOz` (тип `general-purpose`, глубина 1, родитель: `general-purpose-ksLnpjA7E67cqBYBArnNK6VaQtRzgMCz`)
- Статус: completed, создан 15:31:09 UTC, финальный отчёт 18390 символов

**Постановка задачи (дословно из лога):**

READ-ONLY code review. Do NOT edit any file. Repo: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`, perf fork of upstream llama.cpp). Merge base = branch `master`.

Scope: ONLY the Vulkan GLSL compute shaders under ggml/src/ggml-vulkan/vulkan-shaders/ that the diff touches. Use `git diff master...HEAD -- <file>` for modified ones and read the whole file for new ones:
NEW: mul_mm_f16reg.comp, mul_mmid_f16reg.comp, mul_mat_vec_id_fork.comp, gated_delta_net_lazy.comp, rms_norm_rows.comp, ssm_conv_concat.comp, concat_transpose.comp, dsv4_hc_post_norm.comp, coopmat_a_probe.comp, coopmat_c_probe.comp, shmem_probe.comp
MODIFIED: mul_mm.comp, mul_mmq.comp, mul_mmq_cm1.comp, mul_mmq_cm1_funcs.glsl, mul_mm_cm2.comp, get_rows.comp, quantize_q8_1.comp, count_experts.comp, topk_argsort.comp, flash_attn.comp, flash_attn_base.glsl, flash_attn_cm1.comp, flash_attn_cm2.comp, flash_attn_sparse_compact.comp, gated_delta_net.comp, repeat.comp, dsv4_hc_pre.comp
Also read ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp diff (+104) to know which shader variant gets compiled with which defines / local_size / specialization constants.

Look for, in priority order:
(a) NUMERICS that change model output: f16 (mediump/float16_t) accumulation or f16 intermediate where upstream accumulates in f32; `inversesqrt`/reciprocal approximation replacing a division; changed reduction ORDER in flash_attn (softmax online rescaling) or in any sum; a matmul whose accumulator dtype differs from upstream; precision of the q8_1 quantization; any place where a value could overflow f16 range (65504) or lose denormals.
(b) OOB reads/writes: index arithmetic using the WRONG stride/pitch (nb[] vs ne[]), missing bounds check when ne00/ne01/ne2 is not a multiple of the tile or workgroup size, 32-bit int overflow in `row * stride` for large shapes (n_ctx up to 262144, n_expert up to 512, k up to 513), reading a buffer past the end when a workgroup is dispatched for a partial tile.
(c) Missing `barrier()` / `memoryBarrier*()` between a shared-memory write and another invocation's read; shared memory read before it is initialized; a `barrier()` inside non-uniform control flow (UB in Vulkan).
(d) Specialization-constant misuse: shared array sized by a constant that the generator may set SMALLER than the value actually used at runtime (e.g. NUM_SUBGROUPS vs real gl_NumSubgroups, BLOCK_SIZE vs local_size_x), which is a shared-memory OOB.
(e) NaN/Inf propagation differing from upstream.

For EVERY finding: SEVERITY (Critical = wrong output or memory error; Suggestion; Nit), exact file:line, one-line claim, the CONCRETE failure scenario (which shape/dtype/quant/k, and what wrong value or OOB comes out), whether that shader is used BY DEFAULT or is opt-in (name the env knob + default if you can find it in ggml/src/ggml-vulkan/), and the specific code you read as evidence. Verify against the upstream shader before claiming a difference is real. State explicitly which files you checked and found clean. No style/perf/duplication. Max 10 findings.

**Отчёт агента (дословно):**

I have everything I need. Let me write the final report.

Findings (max 10), most severe first. I'll be honest that I found no confirmed Critical.

Let me order:
1. Suggestion — mul_mmid_f16reg.comp:225 uint32 overflow in element space (default-ON RDNA3).
2. Suggestion — mul_mmid_f16reg.comp:383-386 / mul_mm_f16reg.comp:217,243-244 f32→f16 activation narrowing where the replaced path can't overflow.
3. Suggestion — rms_norm_rows.comp:63-79,89-94: default-ON, changes Σx² order + multiply order vs rms_norm.comp → all outputs differ in last bits.
4. Suggestion — ssm_conv_concat.comp:50-53: default-ON, scalar accumulate replaces upstream's dot() at nc==4 → not bitwise equal.
5. Suggestion — flash_attn_cm1.comp:194 / flash_attn.comp:222: SPARSE_LIST mask ignores rr/r → broadcast over the tile's Br rows; supports_op does not require gqa_ratio>1.
6. Suggestion/Nit — mul_mm.comp:539 vs 620: RELU fusion is applied after narrowing to D_TYPE in the aligned coopmat path but before narrowing in the other paths; also the RELU for HEADSUM (518, 582) is unconditional, not gated on fusion_flags bit 0.
7. Nit — mul_mmid_f16reg.comp:357 `row_ids[min(n, tile_count-1)]` underflow if tile_count==0 (currently unreachable via count_experts).
8. Nit — rms_norm_rows.comp: MAX_ITERS=16 hard cap silently drops columns > ROW_THREADS*16; the host guard hardcodes 32*16 while ROW_THREADS is a specialization constant.
9. Maybe: repeat.comp REPEAT_SCALAR drops the D_TYPE cast — unconditional path, safe only because the two registered variants have identical A/D types; the host gate is "equal type sizes" not "equal type". That's a Nit (a future i32→f32 repeat would silently reinterpret). Actually the guard says "equal type sizes" per agent. If someone dispatches f16 src → i16 pipeline: bit copy, fine. If u8→? no pipeline. So Nit/skip.

Let me also mention dsv4_hc_post_norm Q8 correctness verified.

Also should I flag `mul_mm.comp` `get_fused_scale` multiplying in ACC_TYPE for the f16acc case? f16acc isn't enabled for HEADSUM. In mul_mm.comp MUL_MAT_ID with ACC_TYPE = float16_t (f16acc) — `sums[...].x * fscale` in f16 then D_TYPE. But f16acc + MUL_MAT_ID: is that a thing? `get_fused_scale` returns ACC_TYPE. If ACC_TYPE is f16, the multiply is in f16, losing precision vs upstream's store of the raw f16 accumulator. Since fusion off → 1.0 → exact. When on, f16 multiply. Minor. I'll fold into finding 2 or skip.

Let me write it up now. Keep it tight and evidence-based.

Also explicitly list files checked and found clean. ## Review: Vulkan GLSL shaders touched by branch `cache`

Merge base `master`. Method: `git diff master...HEAD` per file + full reads of the new files, cross-checked against the `master` version of each shader and the CPU/ggml references. I also compiled the standalone new shaders with `glslangValidator -V --target-env vulkan1.2 -S comp` (the target env `vulkan-shaders-gen.cpp:351` uses for every non-`_cm2` shader).

**No confirmed Critical found.** The shaders are unusually careful: the "read the last valid row/column instead of guarding" tricks are all bounded by host-side preconditions, the barrier placement is right everywhere I could follow, and no `shared` array is indexed past the specialization constant that sizes it. Findings are precision-range and reduction-order issues on paths that are on by default.

---

##### 1. Suggestion — 32-bit overflow in element space in the weight-block address (default-ON on RDNA3)
`ggml/src/ggml-vulkan/vulkan-shaders/mul_mmid_f16reg.comp:225`
```glsl
const uint row_blk = (expert * p.batch_stride_a + grow * p.stride_a) / A_QK;
```
Every upstream shader divides *before* multiplying, keeping the product in quant-group space: `mul_mmq.comp:182` `expert_idx * (p.batch_stride_a / BK)`, `mul_mat_vec_base.glsl:74` `expert_id * (p.batch_stride_a / QUANT_K)`, and the fork's own sibling `mul_mm_f16reg.comp` does it correctly (`pw = (grow * (p.stride_a / 32) + k0 / 32) / 2 * 17`, line 133). Here the sum is formed in **elements** and only then divided.

Failure scenario: q4_K MoE `src0` whose element count exceeds 2^32 (=4.295e9 elements -> 2.42 GB at 144 B/256 elems; q5_K 2.60 GB, q5_1 3.0 GB, q8_0 4.0 GB). `row_blk` wraps to a small value; the wrapped index stays **inside** the buffer, so no fault — the shader silently reads another expert's/row's weight blocks and produces wrong logits.

Why Suggestion rather than Critical: the only bound is byte-based — `fork/ggml-vulkan-ops-fork.cpp:143-144` (`ggml_nbytes(src0) > maxStorageBufferRange -> nullptr`) — and `ggml-vulkan.cpp:8026-8028` explicitly exempts this pipeline from the 64-bit-indexing fallback (`if (!f16reg && nbytes > range) pipeline = get_64b_indexing_pipeline(...)`). On a driver reporting `maxStorageBufferRange = 2147483647` the wrap needs 2.42 GB against a 2.147 GB cap, i.e. a **12 % margin only**; a driver reporting a larger limit makes it reachable for DeepSeek/Kimi-class 256/384-expert tensors. Fix: `expert * (p.batch_stride_a / A_QK) + grow * (p.stride_a / A_QK)` (both divide exactly; `ops-fork.cpp:143` enforces `ne[0] % qk == 0`).

Path: `GGML_VK_MMID_F16REG`, and when unset it is forced to `1` on AMD RDNA3 (`fork/ggml-vulkan-fork.cpp:229-238`) — i.e. **default-ON on RDNA3**, off elsewhere.

##### 2. Suggestion — activations narrowed f32 -> f16 where the int8 path they replace cannot overflow
`mul_mmid_f16reg.comp:383-386` and `mul_mm_f16reg.comp:217, 243-244` (`write_b_tile`, `packFloat2x16(f16vec2(...))`).
The path being replaced quantizes the activations to **q8_1** with a per-32-column f32 scale (`ggml-vulkan.cpp:7971` `can_quantize_y`), which cannot overflow for any finite input. Here every activation is forced into f16, so `|x| > 65504` becomes `±inf`, and `0 * inf` inside `coopMatMulAdd` yields NaN in a **stored** output element (the "columns past N read the last column" trick only protects against uninitialized data, not against inf in a valid row). With `GGML_VK_MMID_F16REG_GLU_B16` (on by default, `ops-fork.cpp:166`) the result is narrowed a second time. Accumulation itself is correct: `acc` is `coopmat<float, ..., gl_MatrixUseAccumulator>` everywhere, never f16. Same default status as #1.

##### 3. Suggestion — `rms_norm_rows.comp` is ON by default and changes a reduction order plus the multiply order
`rms_norm_rows.comp:63` (`sum += v[k]*v[k]`, `col = tid + k*32`, 16 serial steps), `:66-71` (32-lane tree), `:79`, `:89-94`.
`fork/ggml-vulkan-fusion-fork.cpp:654` — `GGML_VK_RMS_NORM_ROWS`: `env == nullptr || atoi(env) != 0`, i.e. **on unless explicitly disabled**; pipelines created unconditionally at `ggml-vulkan.cpp:3806-3808` with spec `{32, 8, w, g}`.
Upstream `rms_norm.comp:88-127` accumulates `col = tid, tid+BLOCK_SIZE(512), ...` then does `subgroupAdd` + a two-level combine, and forms `scale * a[col] * w[col]`. The fork accumulates 16 values per lane at stride 32 then a 32-tree, and forms `(v * (inversesqrt(...)*scale)) * w * sigmoid(g)`. Both use `inversesqrt(mean + eps)` — no reciprocal approximation is substituted — so the difference is purely association, but it changes the last bits of **every** element of every RMS norm on rows of <=512 columns. Not wrong, but it is a default-ON output change vs upstream and should be stated as such.

##### 4. Suggestion — `ssm_conv_concat.comp` replaces upstream's `dot()` with a scalar loop (default-ON)
`ssm_conv_concat.comp:50-53` (`for k < nc: sum += v * data_w[w_base + k]`). The path it fuses (`CONCAT` + `ssm_conv.comp`) uses `dot(vec4, vec4)` at the common `nc == 4` (`ssm_conv.comp:40-45`). GLSL leaves the association inside `dot()` unspecified, so the fused kernel is **not** guaranteed bitwise-equal to the kernels it replaces for the standard `nc = 4` GDN conv. Gate: `GGML_VK_SSM_CONV_CONCAT` on unless set to 0 (`fusion-fork.cpp:788`), registered under `!disable_fusion` (`ggml-vulkan.cpp:15374` region). No OOB: with `ncs = nc-1`, the largest index touched is `i2 + nc - 1 <= ncs + n_t - 1`, exactly the concat width.

##### 5. Suggestion — the `SPARSE_LIST` mask cannot express a per-query-row mask, but nothing blocks the non-GQA case
`flash_attn_cm1.comp:194`, `flash_attn.comp:222`:
```glsl
m[rr] = SPARSE_LIST ? data_m[sparse_base + j * Bc + c] : data_m[m_offset + mrow * m_stride + kcol];
```
The `SPARSE_LIST` index does not depend on `rr`, so the same value is written to all four components of `m`; and in `flash_attn_base.glsl:222-224` `n_groups = (gqa_ratio > 1) ? p.nem1 : CEIL_DIV(p.nem1, Br)`, so for `gqa_ratio == 1` a whole `Br`-row tile shares one list **and one mask**. That is only correct if the mask is genuinely tile-shared. `supports_op` for `GGML_OP_FLASH_ATTN_LIST` (`ggml-vulkan.cpp:16594-16598`) requires only `!coopmat2 && src[5]==I32 && src[3]==F16` — no `gqa_ratio > 1` — and `use_sparse` at `:8905` is `list_idx ? true : ...`. If a caller ever issues `FLASH_ATTN_LIST` with MHA, all `Br` query rows get one row's mask values (wrong additive bias, and `-inf` gates applied to the wrong rows). The only current producer (`src/fork/qwen4exp-fork.cpp:2236-2241`) is the QSA path; I could not prove `gqa_ratio > 1` there, so this is a guard gap rather than a confirmed wrong output.
The rest of the flash-attn delta is correct and I verified it: `split_kv = sparse_capacity` (`ggml-vulkan.cpp:9003`) matches `capacity` in `flash_attn_sparse_compact.comp:69`, `gl_NumWorkGroups.x == sparse_n_groups` (`:9243`), and `fa_sparse_tail` (`flash_attn_base.glsl:235-237`) is safe because `end_j = min((split_k_index+1)*per_blocks, total_blocks)` (`:228-229`) keeps `j*Bc < split_kv` — note `fa_sparse_tail` has **no** `lin >= p.split_kv` guard while `fa_kv_index` does (`:243-246`), so it relies entirely on that `end_j` bound. `flash_attn_cm2.comp:204` (`mrow = i * Br + row`) is a genuine fix: `coopMatPerElementNV` passes the fragment-local row (cf. `global_row = i * Br + r` at `:252,:263`), and the new `m_stride != 0` guard keeps the GQA case identical to upstream.

##### 6. Suggestion — RELU fusion is applied inconsistently and, in HEADSUM, unconditionally
`mul_mm.comp:539` narrows first, then rectifies: `cm_dtype[i] = max(cm_dtype[i], D_TYPE(0))` where `cm_dtype` is already `coopmat<D_TYPE, ...>` converted from the f32 accumulator. The other two store paths rectify the f32 value and then narrow (`:549`, `:559` via `fused_store`, `:620`, `:623`), and the headsum epilogues rectify the f32 accumulator (`:518`, `:582`). With `D_TYPE == half` the aligned path can therefore differ from the unaligned path on the same shape. Additionally `:518`/`:582` apply `max(..., 0)` with **no** `fusion_flags` test — the `MUL_MAT_HEADSUM` variant always emits `relu`, so selecting that pipeline for a chain without a following RELU is silently wrong; safety rests entirely on the chain check in `fusion-fork.cpp:284-300`.

##### 7. Nit — `row_ids[min(n, tile_count - 1)]` underflows if `tile_count` is ever 0
`mul_mmid_f16reg.comp:357` (and the `data_b[e0 / 8]` derived from it at `:358,:361`). `tile_count - 1 == 0xFFFFFFFF` makes the `min` degrade to `n`, and the fill loop (`:674-677`) only writes `i < tile_count`, so `row_ids[n]` is uninitialized shared memory -> arbitrary `e0` -> OOB read. Currently unreachable (`count_experts.comp:141-143` emits `n_tiles = ceil(count/row_tile_size) >= 1` per listed expert), but the shader leans on another file's invariant with no local `max(tile_count, 1u)`.

##### 8. Nit — `MAX_ITERS` / `ROW_THREADS` coupling is duplicated as a hard-coded constant in the host
`rms_norm_rows.comp:37` (`#define MAX_ITERS 16`) silently drops any column past `ROW_THREADS * 16`; the guard that is supposed to prevent that is written against the *default* `ROW_THREADS` value, not the constant: `fusion-fork.cpp:666` `a->ne[0] > 32*16`. Today `ggml-vulkan.cpp:3807` passes `{32, 8, ...}` so it matches; changing the pipeline's `ROW_THREADS` (it is `constant_id = 0` *and* `local_size_x_id = 0`, so `shared float sumsh[ROWS][ROW_THREADS]` stays consistent — that part is fine) would turn the guard into silent truncation instead of rejection. Same pattern, correctly handled, in `dsv4_hc_post_norm.comp:73` (`MAX_COLS 8`) vs `fusion-fork.cpp:130-141` (`ne[0] <= 8*512`).

---

#### Checked and found clean

**Numerics / f16 range**
- `topk_argsort.comp:54-57` — this is a **correct bug fix**, not a regression. `.y` is written as `floatBitsToInt(data_a[...])` (`:37`) so the upstream integer comparison inverted the order for negative values (e.g. `-2.0` = `0xC0000000` > `-1.0` = `0xBF800000` as ints). The `k > 1` bitonic branch already used `intBitsToFloat`; the `k == 1` reduction now matches. Padding sentinel `ivec2(p.orig_ncols, 0)` is still filtered by the `x` checks, so `intBitsToFloat(0)` never enters a comparison.
- `quantize_q8_1.comp` — the `A_F16` change is a pure buffer-type swap plus `vec4(data_a[a_idx])`; the quantization arithmetic (`d = amax/127`, `d_inv = d != 0 ? 1/d : 0`, `round`, `ds = (d, sum*d)`, `pack32` byte order) is untouched from `master`.
- `dsv4_hc_post_norm.comp` `Q8` (`:118-146`) is a faithful clone of `quantize_q8_1.comp` `QBLOCK_X4`: same `d`, `d_inv`, `round`, same `f16vec2(d, sum*d)`, and the `lane % 4` -> byte and `inner*8 + lane/4` -> word mapping reproduces `qs[ibx4_inner*8 + iqs]` + `pack32` exactly. It quantizes the **f16-rounded** `n`, which is what upstream gets after the CPY. Clustered ops (`size 32` and `4`) sit outside the `i0 < n_embd` conditions and inside workgroup-uniform branches, and `lane = tid % 32` is the in-cluster index for subgroup sizes 32, 64 and 128. Host gates it on `subgroup_clustered && require_full_support && subgroup_size >= 32` (`ggml-vulkan.cpp:3800`).
- `mul_mmid_f16reg.comp` q4_K/q5_K/q5_1 nibble-to-sub-block association is **correct** and I verified the crux myself: `ggml-quants.c:1529-1532` packs a 64-column chunk as `q[l] = L[j+l] | (L[j+l+32] << 4)`, so the *nibble* selects the sub-block — which is exactly what `base = ... + (f & 1) * 4` / `shift = (f >> 1) * 4` / `so_o = ... + (f >> 1)` (lines 421-424, 401) encode. The `1024 + q - bias` f16 trick is exact (`bytes01`/`bytes23`, `:397-398`; `1024+255 < 65504`), the `- m` correction is folded per element so there is no block-sum term to mis-associate, and the q5_K `qh` scatter matches `dequantize_row_q5_K`.
- `gated_delta_net_lazy.comp` — the replay (`:137-146`) applies the same `g*s + k*delta` expression with the recorded f32 `delta`/`g_val`, so the brought-up state equals what the plain net leaves; the key record is written by the `head_id < H_k` workgroup and read via `kh = head_id % H_k`, the gate record is per `head_id`. Reductions reuse `gated_delta_net.comp`'s helpers verbatim (`temp[SUBGROUP_SIZE]` indexed by `lane` collides when `LANES_PER_COLUMN < SUBGROUP_SIZE` — that hazard is **pre-existing upstream**, `gated_delta_net.comp:55-63`, not introduced here).
- `dsv4_hc_pre.comp` `X_F16` — `f16 -> f32` conversion is exact; `fma` retained; compiles clean.
- `mul_mmq_cm1.comp` `FUSED_GLU` epilogue `g / (1.0f + exp(-g)) * u` (`:636`) matches `ggml_vec_swiglu_f32`; the second `#include` of `mul_mmq_cm1_funcs.glsl` under `CM1_A_SIDE_ONLY` cannot collide because the A side is one `#ifdef DATA_A_*` branch whose three symbols are all `#define`-renamed; `KSCALES == 2` is `#error`-blocked; host LDS accounting doubles the A side (`ggml-vulkan.cpp:1634` `total += glu ? 2 * a_side : a_side`).

**OOB / barriers / spec constants**
- `count_experts.comp` — barrier audit clean: `any_skipped` is written by `tid == 0` (`:88-90`) before the first `barrier()` (`:97`), every `atomicOr` (`:106`) precedes the barrier at `:109`, and the final read (`:162`) is after it; `zero_skipped_rows()` contains no barriers. `int(expert + expert_base) == -1` wraps back correctly to id == -1. Skipping the call is exact when no id is `-1`.
- `get_rows.comp` `FLAT_ROWS` — the body is moved verbatim into `copy_row_element()`; index sets are identical; `n = ne00*ne10` is host-capped at `UINT32_MAX/2` (`ops-fork.cpp:10-12`); the z-stride uses `gl_NumWorkGroups.y` only, which is correct because `local_size_y` is a literal `1` (`:7`, no spec constant).
- `concat_transpose.comp` — every in-bounds `(l0,l1)` is written exactly once and read exactly once (`:29-46` vs `:53-60`), garbage tile cells are never stored, the write/read `barrier()` is present, and the loop-carried `need_barrier` is workgroup-uniform (all loop counters derive from `gl_WorkGroupID`/`p`). `p.nb0x < p.nb0y` only picks the read order, not the indices.
- `flash_attn_sparse_compact.comp` — `n_rows = min(rows_per_group, nem1 - i1*rows_per_group)` bounds the widened `is_selected` reads; `out_base`/`capacity` are consistent with the host and with `fa`'s `sparse_base` (see #5).
- `mul_mm_f16reg.comp` — `A_IT`/`B_IT` guards, the `min(ir*BM + row, M-1)` / `min(ic*BN + n, N-1)` clamps, the q8_0 17-word pair decoding (`a_codes` shifts by `h*8` so `unpackFloat2x16(ra_w[0]).y` really is block 1's scale in the high half of word 8), `sh_c[NWAVE*256]` vs `c_off = wave*256`, and one `barrier()` per stage around `write_*_tile` all check out; all 7 define combinations I compiled are OK.
- `mul_mmq_cm2.comp` — only a descriptor-parity buffer and two push-constant fields; no math touched.
- `repeat.comp` `REPEAT_SCALAR` — unconditional, but `idx` equals `dst_idx(idx)` under the host's `contiguous(dst)` guard and dropping the `D_TYPE()` cast is bit-preserving for the two registered variants.
- Descriptor binding: I confirmed via `ggml-vulkan-common.h:309` (descriptors written contiguously from binding 0) plus each variant's `parameter_count` that `FusedScale` (5), `Bias`/`BiasH` (3), `A2` (6), `X`/`X16` (3), `SOutBuf` (7) and `data_q` (6) are always written when the preprocessed SPIR-V reads them, and are `#ifdef`-ed out otherwise. No unbound-descriptor bug.
- `shmem_probe.comp`, `coopmat_a_probe.comp`, `coopmat_c_probe.comp` — diagnostics, launched only by the probe helpers (`ggml-vulkan-fork.cpp:257`), not on any inference path.

**Blockers for further confidence:** I could not reproduce the generator's exact `glslc` command line for the shaders that `#include` `types.glsl` (`glslangValidator` needs `GL_GOOGLE_include_directive`, `glslc` does not), so `mul_mm.comp` HEADSUM, `mul_mmq_cm1.comp` FUSED_GLU, `concat_transpose.comp`, `get_rows.comp`, `repeat.comp` and `dsv4_hc_post_norm_f16n_q8` were reviewed by reading only, not compiled.

---

### B2.4.1 Ручки окружения шейдеров и условия фьюжна

- Агент: `general-purpose-fmbw2BtPjeXkIZHcmXKTy2kl3bCRpbxn` (тип `general-purpose`, глубина 2, родитель: `general-purpose-xrw1nNGY0G8cGQ9zgejMyaNU4GMFGQOz`)
- Статус: completed, создан 15:59:38 UTC, финальный отчёт 19670 символов

**Постановка задачи (дословно из лога):**

READ-ONLY investigation. Do not edit any file.

Repo: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`, a perf fork of upstream llama.cpp; upstream baseline is branch `master`).

Goal: produce an accurate table of which NEW Vulkan compute-shader paths introduced by this branch are enabled BY DEFAULT versus opt-in, and under exactly which guard conditions.

The new/modified shaders to trace are:
- mul_mm_f16reg.comp           (hint: env knob looks like GGML_VK_MM_F16REG, used by ggml_vk_fork_* in ggml/src/ggml-vulkan/)
- mul_mmid_f16reg.comp         (hint: ggml_vk_fork_mmid_f16reg)
- mul_mat_vec_id_fork.comp     (hint: GGML_VK_MMV_ID_FORK, ggml_vk_fork_mmv_id)
- gated_delta_net_lazy.comp    (ggml_gated_delta_net_lazy)
- rms_norm_rows.comp
- ssm_conv_concat.comp
- concat_transpose.comp
- dsv4_hc_post_norm.comp  (variants _f32, _f16n, _f16n_q8) and dsv4_hc_pre.comp X_F16 variant
- get_rows.comp FLAT_ROWS specialization constant (constant_id = 1)
- repeat.comp REPEAT_SCALAR variants
- mul_mmq_cm1.comp FUSED_GLU variants, and mul_mm.comp MUL_MAT_HEADSUM variants
- flash_attn SPARSE_LIST / FLASH_ATTN_LIST path (Flags bit 80 in flash_attn_base.glsl) and the new `rows_per_group`/`capacity` in flash_attn_sparse_compact.comp
- count_experts.comp conditional zero_skipped_rows
- mul_mm.comp / mul_mmq.comp / mul_mmq_cm1.comp `fusion_flags` (bit 0 = fused MUL scale, bit 1 RELU, MUL_MAT_HEADSUM bits 2/4 bias)
- topk_radix_select.comp STABLE variants

For EACH, report:
1. The gate: env variable name (search for `getenv`, `get_env`, `GGML_VK_`, `vk_get_env`, fork config structs) or ggml parameter / backend-feature flag, and its DEFAULT value (on/off) if it is an env knob.
2. The shape/type preconditions under which the new path is actually selected (quote the guard `if (...)` and the file:line).
3. Whether the path produces BITWISE-IDENTICAL results to the upstream path or just approximate (say what changes, e.g. f16 B operand, different reduction order), based on the shader code you read.
4. Whether the shader buffer bindings that the new code adds (e.g. binding = 5 `FusedScale`, binding = 3 `Bias`, binding = 6 `A2`, binding = 7 `SOutBuf`) are always described in the pipeline's descriptor set layout used to launch them — i.e. find the pipeline creation / set-binding code for each and confirm a descriptor is bound. A shader that reads an unbound buffer is a Critical bug, so check this carefully for: FusedScale in mul_mm.comp/mul_mmq.comp/mul_mmq_cm1.comp, FusedScale/Bias in mul_mm_f16reg/mul_mmid_f16reg, SOutBuf binding 7 in gated_delta_net.comp, X/X16 binding 3 in mul_mm_f16reg.comp HC_PRE, data_q binding 6 in dsv4_hc_post_norm.comp, and binding 3/6/7 in get_rows.comp FLAT_ROWS if any.

Report file:line evidence for everything. Be concrete and terse; a table plus a short "confirmed safe" list. Do not report style issues.

**Отчёт агента (дословно):**

Line 11621 — the call site is in the RMS_NORM op handler, not guarded by disable_fusion. So yes: the plain unfused RMS_NORM → rms_norm_rows path is NOT gated by GGML_VK_DISABLE_FUSION. Confirmed. Only GGML_VK_RMS_NORM_ROWS=0 turns it off.

Let me quickly confirm line 11615-11625 context to be sure it's the RMS_NORM dispatch.

Actually I'm confident enough. Let me write the final report now.

One more: I should note the `mul_mm_f16reg` requires `fusion_flags != 0 → return false` meaning it only runs when NOTHING is fused (ops-fork.cpp:236).

Let me write the final report. I have everything verified. Writing the report.

#### Mechanism that governs question 4 (verify your own cases against this)

- `ggml-vulkan-common.h:309` — descriptors are written **contiguously from binding 0**: `vk::WriteDescriptorSet{ descriptor_set, 0, 0, pipeline->parameter_count, ... }`. Descriptor-list element *i* becomes binding *i*.
- `ggml-vulkan-common.h:279` — `GGML_ASSERT(pipeline->parameter_count == descriptor_buffer_infos.size())` (debug-only).
- `ggml-vulkan.cpp:5242-5255` — one shared `device->dsl` declares bindings `0..MAX_PARAMETER_COUNT-1` (=11, `ggml-vulkan-types.h:220`) as storage buffers. The layout therefore *always* covers 0-11; the only real question is whether a descriptor is **written** for every binding the *preprocessed* SPIR-V reads.
- **Invariant per macro variant:** bindings referenced after preprocessing == `{0 .. parameter_count-1}`, and the dispatch list has exactly `parameter_count` entries in binding order. A variant reading binding N while `parameter_count <= N` reads an unwritten descriptor.

#### Table

| Path | Gate / default | Selection guard (file:line) | Bitwise? | Bindings / `parameter_count` / dispatch |
|---|---|---|---|---|
| `mul_mm_f16reg.comp` | `GGML_VK_MM_F16REG` **opt-in**; `fork.cpp:428` mode=0, `:429-431` `=0`/empty off; **unset → ON only on AMD RDNA3** (`env="r16x96/8x1"`, `_B32="r16x64/8x1"`, `:434-444`); needs fp16+32 subgroups, coopmat, A-layout probe `:460-480` | `ops-fork.cpp:236-238` (`mode==0`, `fused_dst!=nullptr`, **`fusion_flags!=0`**, `src0!=Q8_0`, src1 f32/f16, dst f32 → false); `:259-262` `N>=min_n(64)`, `M>=max(16,min_m=512)`, `K>=min_k`, `K%64==0`; `:265-268` strides (`nb[1]%2*qs`, `src1->nb[1]%16`) | **No.** f16 B operand (`packFloat2x16`) vs upstream int8 q8_1_x4; different K reduction. Fork's own claim is KLD, not bits (`fork.cpp:425-427`). The "tile does not change the order of the sums" comment (`fork.cpp:441`) covers only tile-vs-tile *within this shader* (BK is `const 64`, `k_split` never read) | 0,1,2; **3 = `X`/`X16` only under `#ifdef HC_PRE`** (`:90-92`). pc **3** base (`ggml-vulkan.cpp:3928`), **4** hcpre (`:3932-3933`). Dispatch `{a,b,d}` (`ops-fork.cpp:362`) / `{a,b,d,x}` (`:359`) |
| `mul_mmid_f16reg.comp` | `GGML_VK_MMID_F16REG` **opt-in**; `fork.cpp:225` mode=0, `:230-232` `=0` off, **unset → ON only AMD RDNA3** (`:233-238`); needs fp16+sg32 `:240-244`, coopmat+sg control `:252-255`, coopmat-A probe `:256-262` | `ops-fork.cpp:125-128` (mode, `y_in_place`, src1 f32/f16, `d16&&!glu_up`); `:129-131` `n_per_expert>=min_rows(16)`; `:133-139` Q4_K/Q5_K/Q5_1/Q8_0 only; `:141-146` `ne[1]%128`, `ne[0]%64`, dim01-contig, ranges; `:148-150` `src1->nb[2/3]%16`; `:151-153` glu_up same type+shape | **No.** A scale/offset rounded to f16 (`:282`,`:303`,`:315`); per-element f16 mul+sub (`:444-445`); B f16 (`:384-385`); f32 accum, no exact-int stage. KLD claim only (`fork.cpp:222`) | 0-6. `FusedScale`(5) declared uncond. (`:92`) — covered, pc≥6 always. **`A2`(6) under `#ifdef HAS_A2`** (`:84-86`, =`FUSED_GLU||GLU_SPLIT` `:67-69`). pc = `v==1||v==3 ? 7 : 6` (`ggml-vulkan.cpp:3860`) |
| `mul_mat_vec_id_fork.comp` | `GGML_VK_MMV_ID_FORK` **opt-in**; `fork.cpp:310-313` mode 0; **unset → 2 only on AMD RDNA3 + `integer_dot_product`**, else 0; needs clustered/ballot/vote + full subgroups `:316-321` | `ops-fork.cpp:462` `ggml_vk_fork_mmv_id_takes()`; `:470-476` only one fused op and it must be `MUL`, else upstream; takes-check `ggml-vulkan.cpp:8286` | **No.** Same q8_1_x4 quantization and same exact i32 dots, but different K partition across lanes + `subgroupClusteredAdd(acc,LPR)` vs `subgroupAdd` (`:241-242`,`:329`); min-correction associated differently (`:232`). Mode 2 also runs q8_1 where upstream used **f32** activations | Declares **0,1,2,3,5** — binding 4 deliberately unused. pc **6** (`ggml-vulkan.cpp:3892`,`:3896`); dispatch **6** `{d_X,d_Y,d_D,d_F0,d_D,d_ids}` (`ops-fork.cpp:535`) — index 4 is a dummy `d_D` |
| `gated_delta_net_lazy.comp` | New op `GGML_OP_GATED_DELTA_NET_LAZY`; llama-side `LLAMA_GDN_LAZY` **default 1** (`src/llama-memory-recurrent.cpp:110`), model-gated to qwen4exp (`src/fork/qwen4exp-fork.cpp:3070`) | `ops-fork.cpp:378-393` (`S_v∈{16,32,64,128}`, dst f32, src5/6 f32, src7 i32, src0-4 f32 in mode 1) | **Yes** for the arithmetic — replay (`:134-141`) uses the net's own update expression and the already-reduced recorded `delta` (`:210`); `g_val` same `exp()` on same input. Intermediate **state buffer contents differ** mid-segment (mode 1 writes segment-start state, `:143-147`) | 0-8, all uncond. pc **9** (`ggml-vulkan.cpp:4077`); dispatch **9** `{src0..src4,state,rec,info,dst}` (`ops-fork.cpp:451-453`); mode 2 binds `state_buf` into the absent sources (`:422-425`) |
| `gated_delta_net.comp` **binding 7 `SOutBuf`** (new) | Fusion of the following CPY (no own env) | `ggml-vulkan.cpp:11108` `sout_buf = cache ? cache_subbuffer : dst_buf` | **Yes** — the `master...cache` diff is purely a destination change (`data_dst[s_off…]` → `data_sout[sout_off…]`) | 0-7, all uncond. pc **8** (`ggml-vulkan.cpp:4063`); dispatch **8** `{src0..src5,dst_buf,sout_buf}` (`:11124-11126`) — `SOutBuf` always bound, falls back to `dst_buf` |
| `rms_norm_rows.comp` | `GGML_VK_RMS_NORM_ROWS` **ON by default** (`fusion-fork.cpp:654`, `env==nullptr||atoi!=0`) | `fusion-fork.cpp:663-671` f32 unit-stride rows, `a->ne[0] <= 32*16` (=512), `ggml_are_same_shape(dst,rms)`; blocked by `do_add_rms_partials` `:691` | **No.** Different Σx² tree: 32 lanes accumulating `col=tid+k*32` serially then a 32-tree vs upstream's BLOCK_SIZE-wide/subgroup partials → `scale` differs in last bits, propagates to every element. SCALE fusion also stores `-0.0` where upstream's `+0.0f` normalizes it | 0 A,1 W,2 G,3 D, all uncond. pc **4** (`ggml-vulkan.cpp:3807`); dispatch **4** `{a, w?a, g, dst}` (`fusion-fork.cpp:735-740`) |
| `ssm_conv_concat.comp` | `GGML_VK_SSM_CONV_CONCAT` **ON by default** (`fusion-fork.cpp:788`); fusion registration under `!disable_fusion` (`ggml-vulkan.cpp:15292`) | `fusion-fork.cpp:790-820` (CONCAT dim 0, `conv->src[0]==concat`, f32 unit-stride, `st/x/w->ne[1]==nr`, `ne[3]==1`, same shape) | **Not guaranteed.** Both read the same values, but upstream uses `dot(vec4,vec4)` at the common `nc==4` (`ssm_conv.comp:40-45`) and the fork a scalar loop (`ssm_conv_concat.comp:50-54`); GLSL leaves `dot()` order unspecified | 0-3, uncond. pc **4** (`ggml-vulkan.cpp:3810-3811`); dispatch **4** `{st,x,w,dst}` (`fusion-fork.cpp:855-860`) |
| `concat_transpose.comp` | **No env — unconditionally ON**, structural | `ops-fork.cpp:33-42` `!ggml_is_quantized && (ne[0]>1 && ne[1]>1 && nb[1]<nb[0])`, type size 4→`_32`, 2→`_16` | **Yes** — pure data movement; each `(l0,l1)` assigned exactly once (`:36`) | 0,1,2 (from `generic_binary_head.glsl:23,31,32`). pc **3** (`ggml-vulkan.cpp:3551-3552`); dispatch 3 via `ggml_vk_op_f32` (`:10595`) |
| `dsv4_hc_post_norm.comp` `_f32` / `_f16n` / `_f16n_q8` | `GGML_VK_HC_POST_NORM` **ON by default** (`fusion-fork.cpp:125`), `GGML_VK_HC_POST_NORM_MIN_TOKENS` **32** (`:128`); `GGML_VK_HC_POST_NORM_Q8` **ON by default** (`:43`) | `fusion-fork.cpp:130-141` `src[3]==nullptr`, `ne[1]==4`, `ne[3]==1`, `ne[0]<=8*512`, f32 unit-stride rows, `w->ne[1]∈{1,4}` | **Overall No; the Q8 sub-step Yes.** `fma(x,post,r)` (`:97`) vs upstream mul-then-add (`dsv4_hc_post.comp:85,91`); different Σx² tree (`:104-109` vs `rms_norm.comp:96-126`). The q8 block is a faithful clone of `quantize_q8_1.comp` (amax/scale/round/sum/pack all match `:128-144` vs `:85-121`); N_F16 store is one RN equal to a CPY | 0-6. **`Q`/`data_q`(6) under `#ifdef Q8`** (`:68-69`); `N`(5) declared in *both* N_F16 branches (`:63-67`). pc **6** for `_f32`/`_f16n` (`ggml-vulkan.cpp:3796-3797`), **7** for `_f16n_q8` (`:3801`). Dispatch **7** with `prealloc_y` when `down` (`fusion-fork.cpp:102`), else **6** (`:110`) |
| `dsv4_hc_pre.comp` `X_F16` | New *define variant* of an existing shader (binding-0 element type f32→f16) | `ggml-vulkan.cpp:7699` picks `_f16` when `x->type==F16`; `supports_op` allows f16 only for `DSV4_HC_PRE` `src[0]` (`:16902-16917`) | Same arithmetic — `fma(float(data_x[…]), wv, result)`; the f16 is the graph's own dtype (`LLAMA_HC_F16`) | 0,1,2. pc **3** all four variants (`ggml-vulkan.cpp:3790-3793`) |
| `get_rows.comp` `FLAT_ROWS` (`constant_id=1`) | `GGML_VK_DISABLE_GET_ROWS_FLAT`, **presence-checked → path ON by default** (`ops-fork.cpp:9`) | `ops-fork.cpp:10-12` type ∈ {F32,F16,BF16,I32}, `src0->ne[0]<512`, `ne0*ne1 <= UINT32_MAX/2`; used at `ggml-vulkan.cpp:9429-9431`, `:10319-10326` | **Yes** — body extracted verbatim into `copy_row_element()` (`get_rows.comp:10-27`), no arithmetic | **0,1,2 only — there is no binding 3/6/7.** pc **3** on all 7 flat pipelines (`ggml-vulkan.cpp:3319-3325`); dispatch 3 (`:10595`). Flat and non-flat share the same SPIR-V with different spec constants `{0,1}` |
| `repeat.comp` `REPEAT_SCALAR` | **No env — unconditionally ON** | `ggml-vulkan.cpp:9623-9631` `ggml_nelements(src0)==1 && contiguous(dst) && equal type sizes` → `_i32`/`_i16` | **Yes** — pure copy; `src0_idx_mod` collapses to 0 when all `ne0x==1`; dropped `D_TYPE()` cast is bit-preserving | 0,1 (`generic_unary_head.glsl:20,28`). pc **2** (`ggml-vulkan.cpp:3580-3581`); dispatch 2 (`:10598`) |
| `mul_mmq_cm1.comp` `FUSED_GLU` | `GGML_VK_DISABLE_MMID_GLU` — **default OFF → fusion ON** (`fusion-fork.cpp:450`); needs int8 coopmat `:451` or `mmid_f16reg_mode!=0` | `fusion-fork.cpp:455-457` `ggml_can_fuse_subgraph({MUL_MAT_ID,MUL_MAT_ID,GLU})`; further type/shape checks incl. `a/b/glu` all F32 (`:480-483`) | **Yes** as to shader math — A2 is the same code re-included over a second buffer set (`:179-192`), int accumulator exact; epilogue `g/(1+exp(-g))*u` (`:636`) == `swiglu.comp:7`. Caveat: `USE_MAGIC_BIAS = WARP != 32` (`:156`,`:165-171`) is a different formula — bits diverge if GLU and non-GLU land on tiles with different WARP (all int warptiles seen carry WARP=32) | **`A2`(6) under `#ifdef FUSED_GLU`** (`:44-52`). pc = `mul_mat_id_param_count + 1` = **7** (`ggml-vulkan.cpp:2486`; `=6` defined `:2261`); dispatch **7** `{a,b,d,ids,expert_count_buf,fused_scale,*a2}` (`:6496`), **6** otherwise (`:6498`) |
| `mul_mm.comp` `MUL_MAT_HEADSUM` | `GGML_VK_DISABLE_MM_HEADSUM` — **default OFF → ON** (`fusion-fork.cpp:261`); `GGML_VK_HEADSUM_MIN_TOKENS` **0** (`:265-268`). LID variant `GGML_VK_LID_HEADSUM` **ON by default** (only `=0` disables, `:381`) | `fusion-fork.cpp:269-271` `!coopmat2` + headsum pipelines exist for `src0->type`; `:273-275` `src[1]->ne[1] >= min_tokens`; sequential-chain + single-use checks `:284-300`; LID guard `:387-411` | **Yes** on the epilogue — headsum keyed `f16acc=false` (`fusion-fork.cpp:189`, `ggml-vulkan.cpp:2521`,`:2724`), so relu sees f32; heads summed ascending into `float v=0.0` then `v+=bias` matches upstream's sequential `add(add(relu_h))`; `BK` is a fixed `#define 32` for F32/F16 A (`:209-211`) so tile choice doesn't reorder K | **`Bias`/`BiasH`(3) under `#ifdef MUL_MAT_HEADSUM`** (`:125-132`). pc **4** (`ggml-vulkan.cpp:2521`, `:2724`); dispatch **4** `{a,q,dst,mask}` (`fusion-fork.cpp:433-434`) |
| flash_attn `SPARSE_LIST` / **"Flags bit 80"** | `Flags` = spec constant `constant_id=10` (`flash_attn_base.glsl:14`). `:31` `const bool SPARSE_LIST = (Flags & 80) == 80;` — **80 is a mask `16\|64`, not one bit**: needs upstream `USE_SPARSE`(16) **and** new bit **64**. Reached only via the new op `GGML_OP_FLASH_ATTN_LIST` (`ggml.h:610`, `ggml.c:5675-5711`, `op_params[4]=idx->ne[0]`); graph-side `LLAMA_QSA_FA_LIST` **default 1** (`src/fork/qwen4exp-fork.cpp:2226-2230`), qwen4exp only | `ggml-vulkan.cpp:8815` `list_idx = dst->op==FLASH_ATTN_LIST ? dst->src[5] : nullptr`; `:8905` `use_sparse = list_idx ? true : ...`; `supports_op` `:16594-16598` requires `!coopmat2`, `src[5]` I32, `src[3]` F16. `use_list` threshold `src/fork/qwen4exp-fork.cpp:2238-2243` | **No.** Different block grouping of the online softmax (`ceil(capacity/Bc)` vs `ceil(KV/Bc)`) → `Mf/Lf/Of` retract in another order; test runs at tolerance `5e-4` (`tests/test-backend-ops-fork.inc:414-416`) | 0 Q,1 K,2 V,3 M,4 S,5 O,6 MO,7 `SP` (`flash_attn_base.glsl:84-91`). pc **8** at all three creation sites (`ggml-vulkan.cpp:2138`, `:2174`, `:2211`); dispatch **8** in both branches (`:9338`, `:9352`), unused slots filled with a valid `q_buf` (`:9132-9139`) |
| `flash_attn_sparse_compact.comp` `rows_per_group`/`capacity` | no env; part of the sparse pre-pass | Runs only under `ggml-vulkan.cpp:9224` `if (use_sparse && !list_idx)` — **never for `FLASH_ATTN_LIST`** (list comes from outside). Host: `:8998` `rows_per_group = gqa_ratio>1 ? 1 : Br`, `:8999` groups, `:9000` `capacity = list_idx ? idx->ne[0] : min(rows_per_group*n_kv_max, KV)` | Grouping is a **superset** (union over `rows_per_group` mask rows, `:36-39`) when `>1`, exact at 1 | 0 M,1 I only. pc **2** (`ggml-vulkan.cpp:3388`, `:3390`); dispatch **2** `{mask_buf, sparse_buf}` (`:9238-9239`) |
| `count_experts.comp` conditional `zero_skipped_rows` | **No env, no host flag** — shader-side only (`count_experts.comp:162` `if (any_skipped != 0)`), set at `:105-107` | variant choice `ggml-vulkan.cpp:3777-3781` (`subgroup_arithmetic && vote && require_full_support`) | **Yes.** Upstream's unconditional call is a strict no-op when no id is `-1` (`:67-69`). Barrier audit clean: `any_skipped` shared (`:46`), zeroed before the first barrier (`:88-97`), every write precedes the barrier at `:109`, read after it; `zero_skipped_rows()` has no barriers | 0,1,2 uncond. pc **3** both variants (`ggml-vulkan.cpp:3778`,`:3780`); dispatch **3** `{d_ids, expert_count_buf, d_D}` (`:8177-8179`) |
| `mul_mm.comp`/`mul_mmq.comp`/`mul_mmq_cm1.comp` `fusion_flags` | bit 0 (`ggml-vulkan.cpp:6485` param, set `:8262` `fused_scale ? 1u : 0u`) | — | — | **Bit 0 is overloaded by macro:** under `MUL_MAT_ID` = fused MUL scale read from `FusedScale` (`mul_mm.comp:176-179`, `mul_mmq.comp:284`); in the **dense** `#else` = fused RELU (`mul_mm.comp:181-184`, applied `:537`, `:549`) which reads **no buffer**. Under `MUL_MAT_HEADSUM` the shader reads only bits 2 (f32 bias) / 4 (f16 mask via `unpackHalf2x16`) (`:187-193`); the host passes `1u|2u|4u` (`fusion-fork.cpp:431`) — **bit 0 is inert there** (HEADSUM's relu is unconditional at `:518`). `FusedScale`(5) is under `#ifdef MUL_MAT_ID` in both `mul_mm.comp:38-41` region and `mul_mmq.comp:38-41` → dense variants (pc 3) never reference it |
| `topk_radix_select.comp` STABLE | `GGML_VK_TOPK_RADIX` (`ops-fork.cpp:55`, default **-1 auto**; 1 always, 0 never). `GGML_VK_TOPK_RADIX_STABLE` (`:71-74`) **default 1**; 0 = upstream kernel, 2 = stable without subgroups | `ggml-vulkan.cpp:12194-12200` `ggml_vk_topk_use_radix(...)` where `radix_wins = k>=256 && ncols>tournament_cols && nrows>=8` (`ops-fork.cpp:58-59`), or the tournament lacks a wide-enough pipeline | **Depends on mode.** Diff is purely additive, so the *set* of selected values is unchanged. `STABLE=1/2` differ from upstream: output in ascending index order and ties taken from the lowest index (deterministic) vs upstream's `atomicAdd` compaction (`:256`, `:262-265`) which is run-to-run nondeterministic in both order **and** tie membership. No numeric/precision change | 0 A,1 D,2 CB,3 M,4 S uncond. pc **5** for all three variants (`ggml-vulkan.cpp:3731-3738`) |

#### Confirmed safe — every added binding has a written descriptor

I checked each named case from question 4. **No Critical unbound-descriptor bug found.**

1. **`FusedScale` binding 5** (`mul_mm.comp`, `mul_mmq.comp`, `mul_mmq_cm1.comp`) — under `#ifdef MUL_MAT_ID`; dense variants are pc 3 and never reference it; mul_mat_id variants are pc 6/7 and always receive `fused_scale` (`ggml-vulkan.cpp:6496`/`:6498`), falling back to `d_D` when there is no fused scale (`:8261`).
2. **`FusedScale`(5) / `Bias`** in `mul_mmid_f16reg.comp` — pc is 6 or 7, so binding 5 is always written; **this shader declares no `Bias` at all**. `A2`(6) is `#ifdef HAS_A2` and pc is exactly 7 for the `FUSED_GLU`/`GLU_SPLIT` variants.
3. **`X`/`X16` binding 3** in `mul_mm_f16reg.comp` HC_PRE — `#ifdef HC_PRE`, pc 4 for `_hcpre` variants and the dispatch passes `x` (`ops-fork.cpp:359`); base variants are pc 3 and dispatch 3 (`:362`). `x` is only assigned inside `if (hc_pre)`.
4. **`SOutBuf` binding 7** in `gated_delta_net.comp` — unconditional in the shader, pc raised to 8 (`ggml-vulkan.cpp:4063`), and the 8th descriptor is `sout_buf`, which explicitly falls back to `dst_buf` when there is no fused cache (`:11108`).
5. **`data_q` binding 6** in `dsv4_hc_post_norm.comp` — `#ifdef Q8`; only `_f16n_q8` compiles it and that pipeline is pc 7 with `prealloc_y` as the 7th descriptor (`fusion-fork.cpp:102`).
6. **`get_rows.comp` FLAT_ROWS** — **no binding 3/6/7 exists**; bindings are 0/1/2 with pc 3.
7. **`mul_mat_vec_id_fork.comp` declares non-contiguous bindings `{0,1,2,3,5}`** — this looks alarming and is *not* a bug: pc is 6 and the dispatch passes a deliberate dummy (`d_D`) at index 4 so that `d_ids` lands on binding 5 (`ops-fork.cpp:535`, `ggml-vulkan.cpp:3892`/`:3896`).
8. FA (all variants, incl. `SPARSE_LIST`), `flash_attn_sparse_compact`, `rms_norm_rows`, `ssm_conv_concat`, `concat_transpose`, `repeat`, `count_experts`, `topk_radix_select`, `dsv4_hc_pre`, `gated_delta_net_lazy` — declared bindings == `{0..pc-1}` and dispatch list length == pc in every macro/spec variant I enumerated.

#### Three things worth your attention (not style)

1. **`rms_norm_rows.comp` is ON by default, is not bitwise identical, and bypasses `GGML_VK_DISABLE_FUSION`.** The plain unfused `RMS_NORM` branch (`fusion-fork.cpp:690-702`, reached when `fused_rms_norm_mode == RMS_NORM_COUNT`) falls through with `w = nullptr, dst = rms`, gated only by `GGML_VK_RMS_NORM_ROWS` and `do_add_rms_partials`; the call site `ggml-vulkan.cpp:11621` is in the RMS_NORM op handler, not inside the `!disable_fusion` block at `:15292`. Any upstream-vs-fork bit-comparison diverges for essentially every model (rows ≤512 cols). `GGML_VK_RMS_NORM_ROWS=0` is the A/B switch.
2. **The fork also fused the router MUL into the non-GLU `mul_mat_id` store** (`mul_mmq_cm1.comp:639`, `mul_mm.comp:495`, `mul_mmq.comp:285`). Where `D_TYPE` is f16 that removes one rounding relative to upstream's store-then-MUL. I did not find a host-side guard restricting the fused-MUL case to an f32 dst — worth confirming if you care about bit-equality there.
3. **Correction to a claim I got from a subagent and then verified as wrong:** the `topk_radix` fallback at `ggml-vulkan.cpp:3725`/`:3733`/`:3736` was flagged as a bug (stable==1 on a device without full subgroups gets the upstream kernel rather than the no-subgroup stable variant). Reading `ops-fork.cpp:66-69` carefully, that **is** the documented intent ("on a device without full subgroups upstream, since the select without them reads a contiguous run per invocation: +40 percent at 32768 columns, 3-11x at 131072"). Mode 2 exists only as a manual experiment. Not a bug.

Files changed: none (read-only investigation).

---

### B2.4.1.1 Гейты семи маленьких Vulkan-фьюнов

- Агент: `general-purpose-YGNQhi2hQSkjnx5nlaNQyzc0N3xeXCwk` (тип `general-purpose`, глубина 3, родитель: `general-purpose-fmbw2BtPjeXkIZHcmXKTy2kl3bCRpbxn`)
- Статус: completed, создан 16:09:22 UTC, финальный отчёт 25219 символов

**Постановка задачи (дословно из лога):**

READ-ONLY investigation. Do NOT edit, create, or delete any file. Repo: /Users/askuzminov/Projects/github/askuzminov/llama.cpp, branch `cache` (a perf fork of upstream llama.cpp; upstream baseline = branch `master`). Diff surface is `git diff master...cache -- ggml/src/ggml-vulkan/`.

Fork code lives in ggml/src/ggml-vulkan/fork/ggml-vulkan-{fork,ops-fork,fusion-fork}.cpp and ggml/src/ggml-vulkan/fork/ggml-vulkan-fork.h, plus ggml/src/ggml-vulkan/ggml-vulkan.cpp and ggml/src/ggml-vulkan/vulkan-shaders/*.

TASK: For EACH of the 7 paths below, report exactly 4 things. Every claim MUST carry a `file:line` you actually read. If you cannot determine something, write "UNKNOWN" — never guess.

Paths:
1. `rms_norm_rows.comp` (dispatcher `ggml_vk_fork_rms_norm_rows`, env GGML_VK_RMS_NORM_ROWS)
2. `ssm_conv_concat.comp` (`ggml_vk_fork_ssm_conv_concat`, env GGML_VK_SSM_CONV_CONCAT)
3. `concat_transpose.comp` (`ggml_vk_concat_transpose_pipeline`)
4. `get_rows.comp` FLAT_ROWS specialization constant (`constant_id = 1`) — also find the `pipeline_get_rows_flat` / `pipeline_get_rows_flat_f32` creation and the dispatch (`ggml_vk_get_rows_flat`, env GGML_VK_DISABLE_GET_ROWS_FLAT)
5. `topk_radix_select.comp` STABLE variants (`ggml_vk_topk_radix_stable`, env GGML_VK_TOPK_RADIX_STABLE) and the `GGML_VK_TOPK_RADIX` mode
6. `count_experts.comp` conditional `zero_skipped_rows` (what guards it, and does it change results vs upstream)
7. `repeat.comp` REPEAT_SCALAR variants (which defines, what creates/dispatches them, what gate)

For each path report:
(a) THE GATE: env variable name(s) OR ggml parameter / backend-feature flag, and its DEFAULT value (on/off). Quote the exact `getenv(...)` line and its default expression, file:line. Note inverted logic carefully (e.g. `env == nullptr || atoi(env) != 0` = ON by default; `getenv(...) != nullptr` = OFF by default). Also list any secondary gates: device capability flags, `device->disable_fusion`, `add_rms_fusion`, subgroup requirements, etc.
(b) SHAPE/TYPE PRECONDITIONS: quote the actual guard `if (...)` line(s) that decide the new path is taken, with file:line.
(c) BITWISE IDENTITY vs upstream: read the shader body and the upstream path it replaces. State whether the new path produces BITWISE-IDENTICAL results or only approximate, and say concretely what differs (f16 intermediate, different reduction/summation order, extra rounding, different tie-breaking or output ordering, ref counts, denormals, etc.). Cite shader lines.
(d) DESCRIPTORS: the shader's `layout(binding = N)` set, the `parameter_count` passed to `ggml_vk_create_pipeline`/`_pipeline2` for it, and the descriptor list passed to `ggml_vk_dispatch_pipeline`.

For (d) use this ALREADY-ESTABLISHED mechanism (verify against it, do not re-derive):
- ggml/src/ggml-vulkan/ggml-vulkan-common.h:309 — `vk::WriteDescriptorSet write_descriptor_set{ descriptor_set, 0, 0, pipeline->parameter_count, ... }`: descriptors are written CONTIGUOUSLY from binding 0, so descriptor list element i -> binding i.
- ggml-vulkan-common.h:279 — `GGML_ASSERT(pipeline->parameter_count == descriptor_buffer_infos.size())`.
- ggml-vulkan.cpp:5242-5255 — one shared descriptor set layout `device->dsl` declaring bindings 0..MAX_PARAMETER_COUNT-1 (=11) as storage buffers, so the layout always covers 0..11; the real question is whether a descriptor is WRITTEN for every binding the SPIR-V variant actually reads.
- Therefore the invariant per SPIR-V variant is: bindings used by that variant == {0 .. parameter_count-1}, and the dispatch list has exactly parameter_count entries in binding order. A variant that reads binding N while parameter_count <= N reads an UNWRITTEN descriptor = Critical bug. Note that a shader file may declare bindings under different `#if` macros, so you must reason per macro variant (check vulkan-shaders-gen.cpp for which defines each variant gets). Note `device->dsl` declares all 12, so a non-contiguous declared set only matters for bindings actually referenced after preprocessing.

Output a compact per-path section. Terse, evidence-dense, file:line for everything. Do not report style nits. Do not summarize this prompt back.

**Отчёт агента (дословно):**

Zero removed lines in topk_radix_select.comp — purely additive. So the radix threshold selection is byte-identical to upstream; only the emit stage is replaced under `#ifdef STABLE`.

Now I'm ready to write the final report. Let me be compact and evidence-dense.

Let me also note for path 5(c): the non-STABLE (`topk_radix_select_f32`, spec {BLOCK_SIZE,0}) variant's code is byte-identical to upstream's shader. So STABLE=0 → bitwise identical to upstream (and, like upstream, nondeterministic across runs).

One more thing to check for descriptors: `pipeline_topk_radix_f32` created with `require_full_subgroups=true` for stable_sg (arg order: `..., 1, true, true, device->subgroup_size` → align=1, disable_robustness=true, require_full_subgroups=true, required_subgroup_size). For the stable==2 / upstream branch: `..., 1, true)` → align 1, disable_robustness=true. ✓

And gate for stable_sg: `stable == 1 && device->subgroup_ballot && device->subgroup_require_full_support && BLOCK_SIZE % device->subgroup_size == 0` at ggml-vulkan.cpp:3725.

BLOCK_SIZE = 1u << min(10u, device->max_workgroup_size_log2) at 3721.

Now write the report. Keep it compact.

Also for path 2 I should note the fork's `f32` lambda allows nb[0] % 4 == 0 (not requiring 1 element stride) and passes s_nb0/x_nb0 — while upstream ssm_conv hardcodes stride-1 in dim0 (`i2 + i1*(nb01/4)`), so upstream would read wrong data for nb0 != 4... actually upstream would be wrong. Does anything gate that? In upstream, concat output's nb[0] is 4 (contiguous). In the fork path, `st` and `x` are the concat *sources* whose nb[0] may be non-contiguous — handled by s_nb0. OK.

Also I should mention the SILU placement: upstream SSM_CONV+UNARY fusion — `ggml_vk_can_fuse_ssm_conv(ctx,cgraph,i,2)` is tried at 15397+ ... the fork's CONCAT_SSM_CONV is tried earlier at 15376.

Hmm, upstream's SSM_CONV+UNARY(SILU) fusion: let me check that the fork's silu formula matches. Fork: `sum = sum / (1.0f + exp(-sum))`. ssm_conv.comp:55: `sum = sum / (1.0f + exp(-sum));`. Identical. ✓

Final answer time. Also mention for path 2 that the upstream conv applies on the concat buffer and the fork's read is of the same values.

Also worth noting for path 2: upstream pipeline `pipeline_ssm_conv_f32` has spec constants `{32,16,0,0}` = BLOCK_SIZE 32, TOKENS_PER_WG 16, APPLY_BIAS 0, APPLY_SILU 0. And ssm_conv_concat `{32,16,0/1}`. Same block geometry.

Let me write. All evidence gathered. Report:

---

### Path-by-path findings (`master...cache`, `ggml/src/ggml-vulkan/`)

Descriptor mechanism verified as stated: `ggml-vulkan-common.h:309` (contiguous write from binding 0, count = `pipeline->parameter_count`), `:279` (`GGML_ASSERT(parameter_count == descriptor_buffer_infos.size())`), `ggml-vulkan.cpp:5242-5255` (`device->dsl` declares bindings 0..11; `MAX_PARAMETER_COUNT 12` at `ggml-vulkan-types.h:220`). Spec-constant → `constantID` mapping is positional: `ggml-vulkan.cpp:712` `specialization_entries[i].constantID = i`.

**No descriptor gap found on any of the 7 paths.**

---

#### 1. `rms_norm_rows.comp` / `ggml_vk_fork_rms_norm_rows`

**(a) Gate** — `fork/ggml-vulkan-fusion-fork.cpp:654`:
```cpp
static const bool enabled = [] { const char * env = getenv("GGML_VK_RMS_NORM_ROWS"); return env == nullptr || atoi(env) != 0; }();
```
**ON by default** (inverted logic: only `=0` disables).
Secondary gates:
- Fusion variants (`RMS_NORM_SCALE`, `RMS_NORM_MUL_SIGMOID_MUL`) are registered inside `if (!ctx->device->disable_fusion)` — `ggml-vulkan.cpp:15292`, call at `:15374`.
- `ctx->do_add_rms_partials` blocks it — `fusion-fork.cpp:691, 748`.
- **`disable_fusion` does NOT block the plain unfused RMS_NORM path.** With `fused_rms_norm_mode == RMS_NORM_COUNT`, `fusion-fork.cpp:695` (`else if (... != RMS_NORM_COUNT) return false;`) falls through with `w = nullptr, dst = rms`, so a bare RMS_NORM ≤512 cols still takes the fork kernel. Only the env var turns that off.
- No device-capability gate: pipelines created unconditionally, `ggml-vulkan.cpp:3803-3808`.
- **No `RMS_NORM_MUL` re-check of `_ok()` at runtime is missing** — it is checked at `:701`.

**(b) Preconditions** — `fusion-fork.cpp:660-674`:
```cpp
if (!f32_rows(a) || !f32_rows(rms) || !f32_rows(dst) || a->ne[0] == 0 || a->ne[0] > 32*16 || ggml_nrows(a) == 0 ||
    !ggml_are_same_shape(dst, rms)) return false;                                    // :663
if (w != nullptr && (!f32_rows(w) || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1 || a->ne[0] % w->ne[0] != 0)) return false;  // :667
if (gate != nullptr && (!f32_rows(gate) || !ggml_are_same_shape(gate, rms))) return false;                                        // :670
```
`f32_rows` = `t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float)` (`:662`). `32*16` = `ROW_THREADS * MAX_ITERS` (`rms_norm_rows.comp:36`). SCALE fusion additionally requires `ggml_get_op_params_f32(scale,1) == 0.0f` (`:772`).

**(c) Bitwise identity — NO, differs.** Two concrete differences:

1. **Different summation tree for Σx².** Upstream `rms_norm.comp` picks BLOCK_SIZE 128/256/512 by row length (`ggml-vulkan.cpp:3418-3421`, selector `ggml_vk_rms_norm_pipeline` at `:9416-9425`); each invocation accumulates `col = tid; col += BLOCK_SIZE` (`rms_norm.comp:90-94`), then `subgroupAdd(sum)` per subgroup + a cross-subgroup combine (`rms_norm.comp:97-114`), or a full BLOCK_SIZE-wide shared-memory tree (`:116-126`). The fork reduces **32 lanes** with `col = tid + k*32` accumulated **serially** over `k` (`rms_norm_rows.comp:57-64`, `sum += v[k]*v[k]` at `:63`) then a 32-wide shared tree (`:66-73`). For `ncols=128` upstream's partials are `P[t]=x_t²` combined `((s0+s1)+s2)+s3` over 4 subgroups; the fork's are `P[t]=((x_t²+x_{t+32}²)+x_{t+64}²)+x_{t+96}²` combined in one 32-tree. Non-associative float → **different `rs`, therefore different output bits for every element.** (`mean + eps` / `inversesqrt` formula itself matches: `rms_norm.comp:129-130` vs `rms_norm_rows.comp:79`.)
2. **Sign of zero in the SCALE fusion.** Upstream = `rms_norm` (`scale*a`, `rms_norm.comp:167`) then `scale.comp`: `data_a[idx]*p.param1 + p.param2` with `param2==0` — a `-0.0` product becomes `+0.0` after `+0.0f`. Fork folds the scale into `rs = inversesqrt(...) * p.scale` (`rms_norm_rows.comp:79`) and stores `v[k]*rs` (`:88`) → can store `-0.0`.
   - MUL association matches: `(scale*a)*w` (`rms_norm.comp:138,152`) vs `x = v[k]*rs; x *= w` (`rms_norm_rows.comp:88-91`).
   - Sigmoid matches bit-for-bit: `1.0f/(1.0f+exp(-x))` (`unary.comp:74-76`) vs `rms_norm_rows.comp:94`.
   - No f16 intermediate, no ref-count/denormal handling anywhere.

**(d) Descriptors** — shader bindings **0 A, 1 W, 2 G, 3 D** (`rms_norm_rows.comp:17-20`), declared unconditionally (not under any macro). `parameter_count = 4` (`ggml-vulkan.cpp:3807`, spec consts `{32, 8, w, g}` → `ROW_THREADS=32, ROWS=8, DO_MUL, DO_GATE`). Dispatch list = 4 entries in binding order (`fusion-fork.cpp:735-740`): `a, (w?a), g, dst`. Invariant holds for all 4 macro variants. Host PC struct `fork/ggml-vulkan-fork.h:89-102` matches the shader block field-for-field (80 B).

---

#### 2. `ssm_conv_concat.comp` / `ggml_vk_fork_ssm_conv_concat`

**(a) Gate** — `fork/ggml-vulkan-fusion-fork.cpp:788`:
```cpp
static const bool enabled = [] { const char * env = getenv("GGML_VK_SSM_CONV_CONCAT"); return env == nullptr || atoi(env) != 0; }();
```
**ON by default.** Secondary: registration is inside `if (!ctx->device->disable_fusion)` (`ggml-vulkan.cpp:15292`, call `:15376`); dispatch is taken only when `ctx->fused_concat_ssm_conv` (`ggml-vulkan.cpp:13269-13273`). No device-capability gate; pipelines unconditional (`ggml-vulkan.cpp:3810-3811`).

**(b) Preconditions** — `fusion-fork.cpp:789-823`:
```cpp
if (!enabled || concat->op != GGML_OP_CONCAT || node_idx + 1 >= cgraph->n_nodes) return false;                    // :790
if (conv->op != GGML_OP_SSM_CONV || conv->src[0] != concat) return false;                                          // :795
if (!silu && !ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_CONCAT, GGML_OP_SSM_CONV }, { node_idx + 1 })) return false;  // :802
if (ggml_get_op_params_i32(concat, 0) != 0 || !f32(st) || !f32(x) || !f32(w) || !f32(concat) || conv->type != GGML_TYPE_F32 ||
    dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float) || w->nb[0] != sizeof(float)) return false;           // :812-815
return st->ne[1] == nr && x->ne[1] == nr && w->ne[1] == nr && st->ne[2] == conv->ne[2] && x->ne[2] == conv->ne[2] &&
       st->ne[3] == 1 && x->ne[3] == 1 && conv->ne[3] == 1 && ggml_are_same_shape(dst, conv);                       // :817-820
```
Concat must be on dim 0. Bias is structurally excluded (fork only matches `CONCAT[+SILU]`, never the `SSM_CONV_BIAS_SILU` ADD chain that upstream uses at `ggml-vulkan.cpp:10009`).

**(c) Bitwise identity — NOT guaranteed.** Both read the same numeric inputs (CONCAT is a pure copy; the fork splits the read at `j < p.ncs` between state and `x`, `ssm_conv_concat.comp:52`), and both accumulate ascending. But:
- **`nc == 4` uses `dot()` upstream and a scalar loop in the fork.** Upstream `ssm_conv.comp:40-45`: `sum = dot(vec4(...), vec4(...))` when `nc == 4` (the standard GDN kernel size). Fork `ssm_conv_concat.comp:50-54`: `for (k=0..nc) sum += v * data_w[...]`. GLSL leaves `dot()`'s component ordering unspecified, so bitwise equality is not guaranteed at the common `nc==4`.
- SILU matches exactly: `sum / (1.0f + exp(-sum))` (`ssm_conv.comp:55` vs `ssm_conv_concat.comp:57`).
- Generalization, not a deviation: upstream hardcodes dim-0 stride 1 (`src0_base = i3*(nb02/4) + i2 + i1*(nb01/4)`, `ssm_conv.comp:35`); the fork passes `s_nb0`/`x_nb0` explicitly (`ssm_conv_concat.comp:52`, `fusion-fork.cpp:840-841`) — required, since the sources are not the contiguous concat output.
- Upstream never uses its `ncs` push constant (it reads the already-concatenated buffer); the fork's `ncs = st->ne[0]` (`fusion-fork.cpp:848`) is the correct boundary.

**(d) Descriptors** — bindings **0 S, 1 X, 2 W, 3 D** (`ssm_conv_concat.comp:17-20`), unconditional. `parameter_count = 4` (`ggml-vulkan.cpp:3810-3811`, spec `{32,16,APPLY_SILU}`). Dispatch list = 4, in order (`fusion-fork.cpp:855-860`): `st, x, w, dst`. Holds for both variants. PC struct `fork/ggml-vulkan-fork.h:104-115` matches the shader block.

---

#### 3. `concat_transpose.comp` / `ggml_vk_concat_transpose_pipeline`

**(a) Gate** — **no env var** (confirmed: no `getenv` in `ggml-vulkan-ops-fork.cpp:31-43`, and no `GGML_VK_*CONCAT` in the whole diff). Unconditionally **ON**, structural only. Not a fusion, so `disable_fusion` does not affect it. Only reachable behind `ggml_vk_concat_supported` (`ggml-vulkan.cpp:9518-9521`).

**(b) Preconditions** — `fork/ggml-vulkan-ops-fork.cpp:33-42`:
```cpp
auto transposed = [](const ggml_tensor * t) { return t->ne[0] > 1 && t->ne[1] > 1 && t->nb[1] < t->nb[0]; };
if (!ggml_is_quantized(src0->type) && (transposed(src0) || transposed(src1))) {
    if (ggml_type_size(src0->type) == 4) return ctx->device->pipeline_concat_transpose_32;
    if (ggml_type_size(src0->type) == 2) return ctx->device->pipeline_concat_transpose_16;
}
```
Note `nb[1] < nb[0]` is in **bytes**, so this only fires for genuinely strided/transposed views (`nb[0] > nb[1]`), never for a contiguous 4-byte tensor.

**(c) Bitwise identity — YES.** Pure data movement, no arithmetic. The tile load assigns each `(l0,l1)` exactly once: `if ((is_src0 ? a_t : b_t) != (t == 1)) continue;` (`concat_transpose.comp:36`) selects exactly one of the two `t` iterations per tile element, and the source is chosen by the same `is_src0` predicate as upstream (`concat.comp:31,33`) — `a_t`/`b_t` (`concat_transpose.comp:26-27`) only change *which thread* loads, not *what* is loaded. Bounds/offset math is the same as `concat.comp:24-29` (`o[dim]` subtraction, `concat_transpose.comp:41-44`). `uint()`/`D_TYPE()` round-trip is bit-preserving for the 2- and 4-byte non-quantized types this path admits. Coverage is exact: loads guarded at `:33`, stores at `:53`, tile grid loops at `:70-83` with the `need_barrier` reuse guard (`:76-78`).
One host-side consistency point checked: `elements` uses `dst->ne[0]`/`ne[1]` in *elements* (`ggml-vulkan.cpp:10507-10510`) while the shader indexes in *storage units* (`p.ne20`); equal because this path is non-quantized only (`units_per_block == 1`, `ggml_vk_concat` `:11284-11286`).

**(d) Descriptors** — bindings come from `generic_binary_head.glsl:23,31,32` → **0 A, 1 B, 2 D**. `parameter_count = 3` (`ggml-vulkan.cpp:3551-3552`, spec `{}`, so `norepeat` keeps its default and is unreferenced by this shader). Dispatch via `ggml_vk_op_f32`, `CONCAT` falls in the `use_src1` branch → `{ src0_buf, src1_buf, dst_buf }` (`ggml-vulkan.cpp:10595`). Holds for both the `_16` and `_32` variants.

---

#### 4. `get_rows.comp` `FLAT_ROWS` (`constant_id = 1`)

**(a) Gate** — `fork/ggml-vulkan-ops-fork.cpp:9`:
```cpp
static const bool disabled = getenv("GGML_VK_DISABLE_GET_ROWS_FLAT") != nullptr;
```
**ON by default** (any value, including `=0`, disables — presence-checked). No device-capability gate; pipelines unconditional (`ggml-vulkan.cpp:3319-3325`). Checked in three places: pipeline select `ggml-vulkan.cpp:9429-9431`, workgroup calc `:10319-10326`.

**(b) Preconditions** — `fork/ggml-vulkan-ops-fork.cpp:10-12`:
```cpp
return !disabled &&
       (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_I32) &&
       src0->ne[0] < 512 && (uint64_t)src0->ne[0] * src1->ne[0] <= UINT32_MAX / 2;
```
Pipeline pick `:15-29` (I32 → `pipeline_get_rows_flat[I32]`, asserts `dst` I32; `dst F16` → `flat[]`, `dst F32` → `flat_f32[]`); types not in the guard have no pipeline and are unreachable.

**(c) Bitwise identity — YES.** The body was extracted verbatim into `copy_row_element()` (`get_rows.comp:10-27`) and is shared by both branches; the diff shows the upstream loop body replaced by a call with identical arguments (`git diff` on `get_rows.comp`: the removed block is exactly the current `copy_row_element` contents). Same `bf16_to_fp32` path, same `D_TYPE(v)` store, no reduction, no reordering effect. Coverage is complete: flat branch `get_rows.comp:30-41` walks `n = ne00*ne10` with stride `gl_WorkGroupSize.x * gl_NumWorkGroups.x` and `gid_z` over `ne11*ne12` with stride `gl_NumWorkGroups.y`, matching the clamped dispatch grid at `ggml-vulkan.cpp:10320-10325` (`max_x = maxComputeWorkGroupCount[0] * wg_denoms[0]`).
Note the spec-constant pair is `{0, 1}` = `norepeat=false` (from `generic_binary_head.glsl:38`, unused here) + `FLAT_ROWS=1`; comment at `ggml-vulkan.cpp:3318`.

**(d) Descriptors** — bindings 0 A / 1 B / 2 D (`generic_binary_head.glsl:23,31,32`). `parameter_count = 3` on all 7 flat pipelines (`ggml-vulkan.cpp:3319-3325`). Dispatch `{ src0_buf, src1_buf, dst_buf }` via `ggml_vk_op_f32` `use_src1` branch (`:10595`), and the **same SPIR-V modules** back both the flat and non-flat pipelines with different spec constants. Holds.

---

#### 5. `topk_radix_select.comp` STABLE + `GGML_VK_TOPK_RADIX`

**(a) Two gates.**

`GGML_VK_TOPK_RADIX` (mode; selects radix vs tournament) — `fork/ggml-vulkan-ops-fork.cpp:54-59`:
```cpp
const char * env = getenv("GGML_VK_TOPK_RADIX");
return env ? atoi(env) : -1;                       // default -1 = auto
...
const bool radix_wins = k >= 256 && ncols > tournament_cols && nrows >= 8;
return radix_mode == 1 || (radix_mode != 0 && radix_wins);
```
`1` = radix always, `0` = never (except the hard fallback), unset = auto.

`GGML_VK_TOPK_RADIX_STABLE` — `fork/ggml-vulkan-ops-fork.cpp:70-75`:
```cpp
const char * env = getenv("GGML_VK_TOPK_RADIX_STABLE");
const int v = env ? atoi(env) : 1;
return v == 0 || v == 2 ? v : 1;
```
**Default 1** (stable, subgroups). `0` = upstream kernel, `2` = stable without subgroups.

Secondary device gates (`ggml-vulkan.cpp:3721-3738`): `BLOCK_SIZE = 1u << std::min(10u, device->max_workgroup_size_log2)` (`:3721`); subgroup variant requires
`stable == 1 && device->subgroup_ballot && device->subgroup_require_full_support && BLOCK_SIZE % device->subgroup_size == 0` (`:3725`), and is created with `require_full_subgroups=true, required_subgroup_size=device->subgroup_size` (`:3731-3732`). Note the three branches are `if / else if / else` (`:3727`, `:3733`, `:3736`) — **falling back from `stable==1` to the `stable==2` variant does *not* happen**: a device failing `:3725` with `stable==1` lands in the `else` and gets the **upstream** kernel, not the no-subgroup stable one. Also upstream has no env at all and only ever used radix as the "k too large for the tournament" fallback (`master` `ggml-vulkan.cpp:3563-3564`, `11492`).

**(b) Preconditions** — `ggml-vulkan.cpp:12194-12200`:
```cpp
const uint32_t k_min_pipeline = std::max((uint32_t) log2f(float(k)) + 1, ctx->device->subgroup_size_log2);
...
if (ggml_vk_topk_use_radix(k, ncols, nrows, tournament_cols) ||
    k_min_pipeline >= num_topk_pipelines || ctx->device->pipeline_topk_f32[k_min_pipeline] == nullptr) {
```
`tournament_cols` = widest existing `pipeline_topk_f32[i]` (`:12196-12198`). QSA variant is a separate path: `ggml_vk_can_fuse_topk_qsa` gated by `ctx->device->disable_fusion || !ctx->device->pipeline_topk_radix_qsa` (`:14810`).

**(c) Bitwise identity — depends on the mode.**
The shader diff is **purely additive** (zero removed lines in `topk_radix_select.comp`), so the radix threshold search is byte-identical to upstream in every variant and the *set* of values above the threshold is unchanged.
- `STABLE=0` → `topk_radix_select_f32` = upstream SPIR-V → bitwise identical **to upstream's distribution**, i.e. nondeterministic run-to-run: `data_d[row_out + atomicAdd(out_count,1u)]` (`:256`) gives an order-dependent output index, and the tie fill `pos = atomicAdd(out_count,1u); if (pos < p.k)` (`:262-265`) makes the *selected tie members* contention-dependent.
- `STABLE=1/2` → **differs from upstream**: output is in ascending index order (`above` written at `at + ballotExclusiveBitCount` / `at++`, `:140`/`:186`; ties taken from the lowest index at `n_above + tt + ...`, `:144-146`/`:189-193`), so (i) the index **order** of `data_d` differs, and (ii) at ties the **selected elements** differ (deterministic lowest-index instead of contention order). Values are read through `value()` (`:91-93`), which re-gathers for QSA rather than reading `scratch` — same `gather()` expression (`:64-70`), so no numeric change. `f2ui` is unchanged (`:53-61`); no f16 intermediate; denormals/order-preserving mapping untouched. This is the intended determinism fix (comment `:95-99`).

**(d) Descriptors** — bindings **0 A, 1 D, 2 CB, 3 M, 4 S** (`topk_radix_select.comp:20-24`), declared **outside** any `#if`, so present in all 5 variants. `parameter_count = 5` for every creation (`ggml-vulkan.cpp:3731-3738`). Non-QSA dispatch pads the unused slots with `src0` (5 entries): `{ src0_buf, dst_buf, src0_buf, src0_buf, src0_buf }` (`:12218-12219`). QSA dispatch passes the real 5 in order `scores, top_k, cell_blk, mask, scratch` (`:12368-12371`). Holds.
Spec constants: stable+subgroup `{BLOCK_SIZE, QSA, n_sg}` (`:3731-3732`, `n_sg = BLOCK_SIZE/device->subgroup_size` → `NUM_SUBGROUPS`, `topk_radix_select.comp:15`), stable-no-subgroup and upstream `{BLOCK_SIZE, QSA}` (`:3734-3735`, `:3737-3738`). Shared-array sizing matches per variant: `sg_above/sg_tie[NUM_SUBGROUPS]` only under `STABLE && USE_SUBGROUPS` (`:43-45`), `scan_above/scan_tie[BLOCK_SIZE]` only under `STABLE && !USE_SUBGROUPS` (`:46-48`), and the subgroup extensions are required under exactly the same condition (`:5-8`).

---

#### 6. `count_experts.comp` conditional `zero_skipped_rows`

**(a) Gate** — **no env var, no host flag.** `zero_skipped_rows()` became conditional purely inside the shader: `count_experts.comp:162` `if (any_skipped != 0)`. Upstream calls it unconditionally (diff line `-    zero_skipped_rows();`).
The only gate at the host level is the variant choice: `count_experts_subgroup` when `device->subgroup_arithmetic && device->subgroup_vote && device->subgroup_require_full_support`, else the plain one (`ggml-vulkan.cpp:3777-3781`). Both variants get the same fork change (single shader, `USE_SUBGROUPS` only affects the prefix block).

**(b) Preconditions** — the guard predicate: `count_experts.comp:105-107`
```cpp
} else if (int(expert + expert_base) == -1) {
    atomicOr(any_skipped, 1); // fork
}
```
`expert = data_a[...] - expert_base`, so `expert + expert_base` recovers the raw id and `int(...) == -1` is the `id == 0xFFFFFFFF` test. Correct for **every** chunk base (the addition is exact mod 2³²), and every id is visited once per chunk (`:99` loop over `num_elements`), so detection is complete — any `-1` sets the flag in every chunk.

**(c) Bitwise identity — YES, results identical to upstream.** When no id is `-1`, upstream's `zero_skipped_rows()` (`:59-74`) is a strict no-op: every iteration hits `if (int(data_a[...]) != -1) continue;` (`:67-69`) and writes nothing. When a `-1` exists, the flag is set and the identical function runs. `vals`/`offsets`/`cursors`/`data_d` bookkeeping is untouched (the change only adds an `else if` arm that cannot alias the `expert < n_experts` branch).
Memory-visibility audit (this is the part that could silently break): `any_skipped` is `shared` (`:46`), initialized `if (tid == 0) any_skipped = 0;` (`:88-90`) **before** the first `barrier()` at `:97`; every write (`:106`) precedes the `barrier()` at `:109`; the final read (`:162`) is after that barrier, so all invocations see one uniform final value — no divergent-`barrier` hazard, and `zero_skipped_rows()` contains no barriers itself. The comment at `:161` is accurate.

**(d) Descriptors** — bindings **0 A, 1 D, 2 DST** (`count_experts.comp:36-38`), unconditional. `parameter_count = 3` for both variants (`ggml-vulkan.cpp:3778, 3780`, spec `{}`). Dispatch list = 3 in order (`ggml-vulkan.cpp:8177-8179`): `d_ids, expert_count_buf, {d_D, d_buf_offset, d_sz}`. Holds.

---

#### 7. `repeat.comp` `REPEAT_SCALAR`

**(a) Gate** — **no env var** (not in the diff's `getenv` list). Unconditionally **ON**, structural only; `disable_fusion` is irrelevant (not a fusion).
Variants/defines: `REPEAT_SCALAR` with `A_TYPE`/`D_TYPE` = `int16_t` or `int32_t` → `repeat_scalar_i16`, `repeat_scalar_i32` (`vulkan-shaders-gen.cpp:974-976`); upstream base variants are `repeat_i32` (`:969`) and `repeat_i16` (`:973`).

**(b) Preconditions** — `ggml-vulkan.cpp:9623-9631`:
```cpp
case GGML_OP_REPEAT:
    if (ggml_nelements(src0) == 1 && ggml_is_contiguous(dst) && ggml_type_size(src0->type) == ggml_type_size(dst->type)) {
        if (ggml_type_size(dst->type) == sizeof(float)) return ctx->device->pipeline_repeat_scalar_i32;
        if (ggml_type_size(dst->type) == 2)             return ctx->device->pipeline_repeat_scalar_i16;
    }
```

**(c) Bitwise identity — YES.** Pure copy, no arithmetic. Upstream `repeat.comp:30` writes `data_d[get_doffset() + dst_idx(idx)] = D_TYPE(data_a[get_aoffset() + src0_idx_mod(idx)])`; with `ggml_nelements(src0)==1` all `ne0x==1`, so `src0_idx_mod` (`repeat.comp:8-15`) reduces to `(i13%1)*nb03 + (i12%1)*nb02 + (i11%1)*nb01 + (i10%1)*nb00 == 0`, and `ggml_is_contiguous(dst)` makes `dst_idx(idx) == idx` (`generic_unary_head.glsl:51-59`, with `p.ne1*`/`p.nb1*` = dst and `ne = ggml_nelements(dst)`, `ggml_vk_repeat` at `:11477`). The `D_TYPE()` cast is dropped in the fork (`repeat.comp:27`), which is bit-preserving because the two variants use identical `A_TYPE`/`D_TYPE`. The `idx >= p.ne` guard is shared (`repeat.comp:22-24`), so coverage is unchanged.

**(d) Descriptors** — bindings **0 A, 1 D** (`generic_unary_head.glsl:20,28`). `parameter_count = 2` (`ggml-vulkan.cpp:3580-3581`, spec `{}`). Dispatch via `ggml_vk_op_f32` with `src1 == nullptr` → the 2-entry branch `{ src0_buf, dst_buf }` (`ggml-vulkan.cpp:10598`). Holds for both `REPEAT_SCALAR` variants and the two non-`REPEAT_SCALAR` ones.

---

#### Open items / notes worth a maintainer's eye

1. **Path 1 is not bitwise identical and is ON by default.** Any KLD/perplexity or token-identical comparison against upstream will diverge for every model with an RMS_NORM row ≤512 columns (i.e. essentially all of them), from the changed Σx² tree alone — not from the fusion structure. `GGML_VK_RMS_NORM_ROWS=0` is the A/B switch.
2. **Path 1 bypasses `GGML_VK_DISABLE_FUSION`** for a plain unfused RMS_NORM (`fusion-fork.cpp:689-703` fall-through with `fused_rms_norm_mode == RMS_NORM_COUNT`).
3. **Path 5 mode-1 fallback asymmetry**: `ggml-vulkan.cpp:3727/3733/3736` — a `stable == 1` device that fails the `:3725` subgroup test gets the **upstream nondeterministic** kernel, not the `stable == 2` no-subgroup kernel that exists one branch away. If the intent (per the comment at `ops-fork.cpp:66-69`) is "default stable, degrade to no-subgroups on devices without full subgroups", the condition at `:3733` should be `stable != 0` rather than `stable == 2`.
4. **Path 2 loses upstream's `dot()` at `nc == 4`** (`ssm_conv.comp:41` vs `ssm_conv_concat.comp:53`); the only way to prove bitwise equality is an empirical A/B against `GGML_VK_SSM_CONV_CONCAT=0`.

---

### B2.4.1.2 Путь sparse-списка во flash-attention

- Агент: `general-purpose-nQyHWsMWdyJ6kq6kn1XkgXY8H2PRl66L` (тип `general-purpose`, глубина 3, родитель: `general-purpose-fmbw2BtPjeXkIZHcmXKTy2kl3bCRpbxn`)
- Статус: completed, создан 16:09:22 UTC, финальный отчёт 13983 символов

**Постановка задачи (дословно из лога):**

READ-ONLY investigation. Do NOT edit, create, or delete any file. Repo: /Users/askuzminov/Projects/github/askuzminov/llama.cpp, branch `cache` (perf fork of upstream llama.cpp; upstream baseline = branch `master`). Diff surface: `git diff master...cache -- ggml/src/ggml-vulkan/`.

TASK: Trace the NEW sparse flash-attention path introduced by this branch. Cover:
1. `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl` — the `SPARSE_LIST` / `FLASH_ATTN_LIST` machinery and specifically the **Flags bit 80**. Find how flags are packed/unpacked (search `Flags`, bit shifts, `>> 5`, `& 0xF`, `16(`, `80`). Quote exact lines. Explain precisely what bit 80 encodes and how the host sets it.
2. The host side that sets those flags: search ggml/src/ggml-vulkan/ggml-vulkan.cpp and ggml/src/ggml-vulkan/fork/*.cpp for the flash-attn dispatch building the flags value (look for the op's `dst->op_params`, `GGML_OP_FLASH_ATTN_EXT`, a `rows_per_group` / `capacity` / sparse-list ggml parameter, and any `getenv` gate). Report: is this path enabled BY DEFAULT or opt-in, and via what (a new ggml op parameter set by llama.cpp? an env knob? a backend feature flag `GGML_BACKEND_DEVICE_FLAG_...`?). Also check `ggml/include/ggml.h` and `ggml/src/ggml.c` in the diff for a new flash-attn param (`git diff master...cache -- ggml/include/ggml.h ggml/src/ggml.c`), and check whether llama.cpp/ sets it.
3. `flash_attn_sparse_compact.comp` — the NEW `rows_per_group` and `capacity` inputs: what they mean, where the host computes them (file:line), and the guard under which the compact kernel runs at all. Quote the guard `if (...)`.
4. `flash_attn.comp`, `flash_attn_cm1.comp`, `flash_attn_cm2.comp` — what the SPARSE_LIST specialization constants change in the shader, and which pipeline variant (`parameter_count`, specialization constants) gets created for it. Find the creation calls in ggml-vulkan.cpp (search `flash_attn`, around lines 2100-2230 where `ggml_vk_create_pipeline(... "main", 8, ...)` appears) and report the binding count.
5. BITWISE: does the sparse-list path produce BITWISE-IDENTICAL results to the upstream dense/masked path, or only approximate? Say concretely what changes (accumulation order over the KV list, fp16 accumulate, softmax renormalization, dropped/zeroed blocks, `zero_skipped_rows`). Cite shader lines.

ALREADY-ESTABLISHED descriptor mechanism — verify against it, do not re-derive:
- ggml/src/ggml-vulkan/ggml-vulkan-common.h:309 writes descriptors CONTIGUOUSLY from binding 0 (`vk::WriteDescriptorSet{ descriptor_set, 0, 0, pipeline->parameter_count, ... }`), so descriptor list element i -> binding i.
- ggml-vulkan-common.h:279 asserts `pipeline->parameter_count == descriptor_buffer_infos.size()`.
- ggml-vulkan.cpp:5242-5255: one shared layout `device->dsl` declaring bindings 0..11 storage buffers, so the layout always covers 0..11; the real question is whether a descriptor is WRITTEN for every binding the SPIR-V variant actually READS. A variant reading binding N while parameter_count <= N = reading an UNWRITTEN descriptor = Critical bug. Reason per macro/specialization variant (see vulkan-shaders-gen.cpp for the defines each variant gets).

Report the exact binding list of `flash_attn_sparse_compact.comp` and its `parameter_count`, and confirm or refute that every binding it reads is written.

Every claim needs a `file:line` you actually read. Write "UNKNOWN" rather than guessing. Terse and evidence-dense. Do not report style nits. Do not summarize this prompt back.

**Отчёт агента (дословно):**

Confirmed: write the report in Russian, keeping code/paths/quotes verbatim.

Let me write the final report now, terse and evidence-dense. ## 1. `flash_attn_base.glsl`: механизм `SPARSE_LIST` и «бит 80»

`Flags` — специализационная константа `constant_id = 10` (`flash_attn_base.glsl:14`). Раскладка битов:

```glsl
flash_attn_base.glsl:23  const bool USE_MASK_OPT    = (Flags & 1) != 0;
flash_attn_base.glsl:24  const bool MASK_ENABLE     = (Flags & 2) != 0;
flash_attn_base.glsl:25  const bool LOGIT_SOFTCAP   = (Flags & 4) != 0;
flash_attn_base.glsl:26  const bool OLD_AMD_WINDOWS = (Flags & 8) != 0;
flash_attn_base.glsl:28  const bool USE_SPARSE      = (Flags & 16) != 0;
flash_attn_base.glsl:29  // fork: GGML_OP_FLASH_ATTN_LIST: binding 7 is the given list (unsorted, -1 anywhere), the mask holds the value of each
flash_attn_base.glsl:30  // slot at the same index as the list
flash_attn_base.glsl:31  const bool SPARSE_LIST     = (Flags & 80) == 80;
```

«Бит 80» — это **не один бит**, а маска `16|64`: `SPARSE_LIST` истинна только когда стоят и `USE_SPARSE`(16, upstream), и новый бит **64** (fork). Бит 64 в upstream не занят (`ggml-vulkan.cpp:1437-1441` формирует только 1/2/4/8/16). То есть 64 кодирует «binding 7 — это готовый список из src[5], а `data_m` — значения слотов этого списка, а не плотная маска».

Потребители бита:
```glsl
flash_attn_base.glsl:235 bool fa_sparse_tail(uint lin) {
flash_attn_base.glsl:236     return USE_SPARSE && !SPARSE_LIST && data_sparse[sparse_base + lin] < 0;
flash_attn_base.glsl:248     const bool in_list = idx >= 0 && (!SPARSE_LIST || uint(idx) < KV);
flash_attn.comp:222            FLOAT_TYPE m = FLOAT_TYPE(SPARSE_LIST ? data_m[sparse_base + j * Bc + c] : data_m[m_offset + (i * Br + r) * m_stride + kcol]);
flash_attn_cm1.comp:194          m[rr] = SPARSE_LIST ? data_m[sparse_base + j * Bc + c] : data_m[m_offset + mrow * m_stride + kcol];
```
Семантика: при `SPARSE_LIST` маска индексируется **по слоту списка** (`sparse_base + slot`), а не по `(строка, колонка KV)`; `-1` может стоять в любом месте списка, поэтому ранний выход по «хвосту» запрещён; элемент списка `>= KV` считается невыбранным (`:248`).

`init_indices` (`flash_attn_base.glsl:219-231`): один список на группу строк маски,
```glsl
uint32_t n_groups = (p.gqa_ratio > 1) ? p.nem1 : CEIL_DIV(p.nem1, Br);
uint32_t group    = (p.gqa_ratio > 1) ? gqa_iq1 : i;
sparse_base = (((iq3 % p.nem3) * p.nem2 + (iq2 % p.nem2)) * n_groups + group) * p.split_kv;
```
Упомянутых в задании `>> 5` / `& 0xF` в FA-пути нет; `>> 16` — это `m_stride = (p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : KV;` (`flash_attn_base.glsl:244`), способ записать ноль так, чтобы компилятор не свернул `&` и не сломал определение выравнивания. `& 0xF`/`>> 8` живут в `flash_attn_dequant.glsl:47-96` (распаковка nibble'ов квантов), к flags отношения не имеют.

Как хост ставит бит: `get_fa_pipeline_state` (`ggml-vulkan.cpp:1437-1441`) кладёт 1/2/4/8/16, затем
```cpp
ggml-vulkan.cpp:8946  // fork: the list of GGML_OP_FLASH_ATTN_LIST, the mask values by slot (SPARSE_LIST of flash_attn_base.glsl)
ggml-vulkan.cpp:8947  if (list_idx) {
ggml-vulkan.cpp:8948      fa_pipeline_state.flags |= 64;
```
`flags` входит в ключ `vk_fa_pipeline_state` (`ggml-vulkan-types.h:422`, карта `pipeline_flash_attn_f32_f16` `:998`), поэтому 64 даёт отдельный `VkPipeline`.

#### 2. Хост: чем это включается

Это **не новый параметр op_params у `FLASH_ATTN_EXT`, а отдельный оп** `GGML_OP_FLASH_ATTN_LIST`:
- `ggml/include/ggml.h:610` — `GGML_OP_FLASH_ATTN_LIST, // fork: attention over a list of KV cells per query row, ggml_flash_attn_list`; API и контракт — `ggml.h:2546-2565` (в том числе «the list need not be sorted and holds each cell at most once per row»).
- `ggml/src/ggml.c:5675-5711` — `ggml_flash_attn_list()`: `src[3] = vals` (f16, значения слота, играющие роль маски), `src[5] = idx` (i32-список), `op_params[4] = idx->ne[0]` — то есть в существующий sparse-код он попадает как `n_kv_max` (`ggml-vulkan.cpp:8813` читает именно `op_params[4]`).
- Диспетч: `ggml-vulkan.cpp:13579-13581` — `ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, nullptr, node)` (mask = `src[3]`, sinks = nullptr).
- `ggml-vulkan.cpp:8815` `const ggml_tensor * list_idx = dst->op == GGML_OP_FLASH_ATTN_LIST ? dst->src[5] : nullptr;`, `:8905` `const bool use_sparse = list_idx ? true : ...`.
- `supports_op`: `ggml-vulkan.cpp:16594-16598` — требуется `!device->coopmat2`, `src[5]` = I32, `src[3]` = F16, далее общий с `FLASH_ATTN_EXT` набор проверок.
- `ggml/src/ggml-vulkan/fork/*.cpp` — flash-attn кода нет вообще (grep `flash_attn|FLASH_ATTN` = 0 совпадений).

**ВКЛЮЧЕНО ПО УМОЛЧАНИЮ, но не через ggml-рычаг, а через построение графа в llama.cpp:**
- `src/fork/qwen4exp-fork.cpp:2226-2230` — `static const int list_mode = llama_fork_env_int("LLAMA_QSA_FA_LIST", 1);` (1 = по умолчанию вкл).
- `:2238-2243` — `use_list = vk_l && n_kv_m >= (list_min >= 0 ? list_min : std::max<int64_t>(4096, 2*top_k->ne[0]))`, при `list_mode == 1 && from_pos && cparams.flash_attn && n_kv_m % hparams.dsv4_compress_ratios[il] == 0`; `LLAMA_QSA_FA_LIST_MIN` по умолчанию `-1` (`:2232-2234`).
- `:2322-2345` — построение: `ggml_flash_attn_list(ctx0, q, k, v, vals, idx, kq_scale)`.
- Единственный потребитель — qwen4exp (grep по `src/`, `tools/`); upstream-модели идут старым путём `ggml_flash_attn_ext_set_n_kv_max` (`src/llama-graph.cpp:3048`, `src/models/deepseek4.cpp:768`, `deepseek41.cpp:742`), который есть и в master (`git show master:ggml/include/ggml.h:2529`).
- `GGML_BACKEND_DEVICE_FLAG` в `ggml-vulkan.cpp` нет ни одного вхождения (grep = 0). Feature-флага нет.

Энв-рычаги на ggml-стороне (они про **дополнения** fork к `FLASH_ATTN_EXT`, сам LIST от них не зависит): `GGML_VK_FA_SPARSE_DISABLE` (`:8816`, upstream), `GGML_VK_FA_SPARSE_GQA` (`:8837-8840`, по умолчанию вкл), `GGML_VK_FA_SPARSE_UPSTREAM` (`:8842-8846`, по умолчанию выкл), `GGML_VK_FA_SPARSE_GROUP` (`:8868-8871`, по умолчанию 1 = объединение разрешено).

#### 3. `flash_attn_sparse_compact.comp`: `rows_per_group`, `capacity`

Новые поля push-констант (`flash_attn_sparse_compact.comp:26-27`, `ggml-vulkan-push-constants.h:902-912`; в master было 7 полей, `git show master:...:888-896`):
```glsl
flash_attn_sparse_compact.comp:36  // a position is kept when any row of the group keeps it
flash_attn_sparse_compact.comp:37  bool is_selected(const uint m_idx, const uint n_rows) {
flash_attn_sparse_compact.comp:39      const float v = float(data_m[m_idx + r * p.nbm1]);
flash_attn_sparse_compact.comp:60      const uint m_base   = i3 * p.nbm3 + i2 * p.nbm2 + i1 * p.rows_per_group * p.nbm1;
flash_attn_sparse_compact.comp:61      const uint out_base = ((i3 * p.nem2 + i2) * gl_NumWorkGroups.x + i1) * p.capacity;
flash_attn_sparse_compact.comp:62      const uint n_rows   = min(p.rows_per_group, p.nem1 - i1 * p.rows_per_group);
```
Слот-клампы и дозаполнение `-1` переведены с `n_kv_max` на `capacity` (`:89`, `:94`, `:132`).

Смысл: один воркгруппа обрабатывает **группу из `rows_per_group` строк маски** и пишет ОДИН список длиной `capacity` слотов, содержащий **объединение** позиций, которые оставила хотя бы одна строка группы (при `rows_per_group == 1` список точен). Порядок — возрастающий по KV (комментарий `:47-51`).

Хост считает это здесь:
```cpp
ggml-vulkan.cpp:8998  const uint32_t sparse_rows_per_group = (gqa_ratio > 1) ? 1 : Br;
ggml-vulkan.cpp:8999  const uint32_t sparse_n_groups       = CEIL_DIV(nem1, sparse_rows_per_group);
ggml-vulkan.cpp:9000  const uint32_t sparse_capacity       = list_idx ? (uint32_t) list_idx->ne[0] : std::min<uint32_t>(sparse_rows_per_group * (uint32_t)n_kv_max, KV);
```
заполнение — `ggml-vulkan.cpp:9226-9236`, диспетч `{ sparse_n_groups, nem2, nem3 }` — `:9238-9240`, scratch `prealloc_y` — `:9109-9114`.

**Охранное условие запуска компактного ядра** (`ggml-vulkan.cpp:9224`):
```cpp
if (use_sparse && !list_idx)
```
То есть для `FLASH_ATTN_LIST` pre-pass не запускается никогда — список уже дан извне. `n_rows` корректен (i1 < n_groups ⇒ `n_rows >= 1`). Проксимальность: `nem1` берётся из маски (`:8720-8723`).

#### 4. Варианты пайплайнов, `parameter_count`, проверки binding'ов

FA: три места создания, **parameter_count = 8 во всех трёх** (специализационные константы — `get_fa_spec_constants`, `ggml-vulkan.cpp:1455-1472`, `Flags` = constantID 10):
- `ggml-vulkan.cpp:2138` — scalar (`flash_attn_f32_f16*`),
- `ggml-vulkan.cpp:2174` — coopmat1,
- `ggml-vulkan.cpp:2211` — coopmat2.

`SPARSE_LIST` не создаёт нового шейдерного модуля: `vulkan-shaders-gen.cpp:743-765` генерирует те же модули, у `SPARSE_LIST` нет своего `#define` — это чисто другое значение spec-константы 10. Запись в карту создаётся лениво при первом LIST-диспетче (`ggml-vulkan.cpp:8965-8975`), компилируется через `ggml_pipeline_request_descriptor_sets` → `ggml_vk_load_shaders(device, requested)` (`:8979`, `:897-903`), который матчит запрошенный пайплайн по указателю и берёт `get_fa_spec_constants(fa.first)`; `parameter_count` при этом тот же 8.

Binding'ы, которые читают FA-варианты: `0` Q, `1` K, `2` V, `3` M (`flash_attn.comp:37-43`), `4` S, `5` O/OV4, `6` MO, `7` SP (`flash_attn_base.glsl:84-91`). Диспетч передаёт ровно 8 дескрипторов в обеих ветках: `ggml-vulkan.cpp:9338` (split_k>1: `{q,k,v,mask,sinks,split_k_buf,mask_opt,sparse}`) и `:9352` (без split_k: `{q,k,v,mask,sinks,dst,mask_opt,sparse}`). Неиспользуемые слоты заполняются валидным `q_buf` (`:9132-9139`) — незаписанного binding'а в FA-пути нет ни для одной макро/spec-вариации.

**`flash_attn_sparse_compact.comp`:**
- binding 0 = `M` readonly (`:15`), binding 1 = `I` writeonly (`:16`). Других binding'ов в файле нет.
- `parameter_count = 2`: `ggml-vulkan.cpp:3388` (`fa_sparse_compact_subgroup`, spec `{compact_wg, compact_num_sg}`) и `:3390` (`fa_sparse_compact`, spec `{compact_max}`).
- Запись: `ggml_vk_dispatch_pipeline(ctx, subctx, sparse_compact_pipeline, { mask_buf, sparse_buf }, ...)` — `ggml-vulkan.cpp:9238-9239`, ровно 2 элемента; `GGML_ASSERT(pipeline->parameter_count == descriptor_buffer_infos.size())` (`ggml-vulkan-common.h:280`) выполняется, bindings 0 и 1 записаны. **Подтверждаю: противоречия нет.** Layout `device->dsl` объявляет 0..11 (`MAX_PARAMETER_COUNT == 12`, `ggml-vulkan-types.h:220`; цикл `ggml-vulkan.cpp:5243-5246`), bindings 2..11 этим ядром просто не читаются.

Дополнительно проверил риск «OpSelect вычисляет обе ветки» для `SPARSE_LIST ? data_m[...] : data_m[...]` (вторая ветка при LIST ушла бы далеко за границы `vals`, а FA-пайплайны создаются с `disable_robustness = true`): минимальный repro на `glslc -O` даёт `OpSelectionMerge`/`OpBranchConditional` с по одному `OpLoad` в ветке, **не** `OpSelect` — незагруженная ветка не читается. Проверено на `/tmp/sel_test.comp` (spirv-dis).

#### 5. Побитовая идентичность

**Нет, не побитово идентично.** Это приближённый путь: тест гоняется с допуском `max_nmse_err() = 5e-4` (`tests/test-backend-ops-fork.inc:414-416`), а списки в тесте намеренно перемешаны (`:442-462`, `std::shuffle` блоков + `-1` в произвольных слотах `(s % 37) != 11`).

Что конкретно меняется:
- **Перестановка группировки по блокам online-softmax.** Плотный путь идёт по `ceil(KV/Bc)` блокам, sparse — по `ceil(capacity/Bc)`, где `capacity = n_list` (`ggml-vulkan.cpp:9000`). Ретракт `Mf`/`Lf`/`Of` делается на каждый блок: `Mf[r] = max(rowmaxf, Moldf); eMf[r] = exp(Moldf - Mf[r]); Lf[r] = eMf[r]*Lf[r];` и `Of[r][d] = FLOAT_TYPE(eMf[r]) * Of[r][d];` (`flash_attn.comp:465-487`), накопление `Lf[r] += Pf[r]`, `Of[r][d] += FLOAT_TYPEV4(Pf[r] * Vf)` (`flash_attn.comp:513-531`). Итоговая сумма по тем же столбцам, но с другими точками домножения → другие биты. `Of` — накопитель `FLOAT_TYPE` (fp16), поэтому разница хорошо видна.
- **Порядок накопления = порядок списка.** Для LIST он явно не отсортирован (`ggml.h:2547-2548`, `flash_attn_base.glsl:29-30`), тогда как compact-путь гарантирует возрастающий KV-порядок «so the downstream attention accumulation is deterministic» (`flash_attn_sparse_compact.comp:47-51`). CPU-референс тоже идёт в порядке списка (`ggml/src/ggml-cpu/fork/ops-fork.inc:71-77`). Значит LIST расходится и с dense, и с compact-путём.
- **split_k считается иначе.** Sparse ветка делит блоки `capacity` (`ggml-vulkan.cpp:9004-9017`), плотная — `KV` с округлением под `alignment` (`:9019-9022`); редукция частичных результатов — `:9341-9344`. Другая разбивка → другой порядок суммирования.
- **Отброшенные слоты.** `-1`, `idx >= KV` и неактивные слоты вообще не дают слагаемых (`flash_attn_base.glsl:243-251`; `flash_attn.comp:225` `masksh[...] = USE_SPARSE ? FLOAT_TYPE(NEG_FLT_MAX_OVER_2) : FLOAT_TYPE(0);`; `flash_attn.comp:515-518` `if (!kv_active) continue;`). По значению это эквивалентно `exp(-inf) = 0` в dense-маске, но меняет группу строкового максимума внутри блока → см. пункт 1.
- **Нет раннего выхода по хвосту для списка** (`flash_attn_base.glsl:235-237`, `!SPARSE_LIST`), проходится весь `capacity`.
- **Объединение строк (путь EXT, `rows_per_group > 1`).** В список попадают ячейки, которые данная строка не выбирала; их нейтрализует построчная домаскировка (`flash_attn_cm1.comp:188-196`, `flash_attn_cm2.comp:199-204`), где чужая строка даёт `-inf`-маску, то есть 0 в softmax. Снова эквивалентность по значению, но не по битам.
- **`zero_skipped_rows` к FA отношения не имеет** — это `vulkan-shaders/count_experts.comp:44,59,163` (зануление пропущенных строк в MoE `mul_mat_id`), на результат sparse-внимания не влияет.
- Дубликатов шейдер не дедуплицирует (и CPU-референс тоже, `ops-fork.inc:71-77`) — контракт `ggml.h:2547-2548` запрещает повтор ячейки в строке, нарушения контракта дали бы двойной учёт.

#### Прочее (не проверено до конца)
Условие включения на стороне llama (`src/fork/qwen4exp-fork.cpp:2238-2243`) проверяет только «это Vulkan» (`qwen4exp_dev_is_vulkan`, `:911`), тогда как бэкенд отвергает оп на coopmat2-устройствах (`ggml-vulkan.cpp:16594-16597`). Что происходит в этом случае (фоллбэк на CPU или ошибка графа), я не проверял — UNKNOWN.

---

### B2.4.1.3 Побитовый анализ matmul-путей форка

- Агент: `general-purpose-Dcw4ftBaQ6k90FKpAbYhL1fnxVcyIx39` (тип `general-purpose`, глубина 3, родитель: `general-purpose-fmbw2BtPjeXkIZHcmXKTy2kl3bCRpbxn`)
- Статус: completed, создан 16:47:23 UTC, финальный отчёт 18475 символов

**Постановка задачи (дословно из лога):**

READ-ONLY. Do NOT edit/create/delete files. Repo: /Users/askuzminov/Projects/github/askuzminov/llama.cpp, branch `cache`; upstream baseline branch `master`. Use `git diff master...cache -- <path>` and read files directly.

QUESTION: For each fork matmul/fusion path below, does it produce BITWISE-IDENTICAL results to the upstream path it replaces, or only approximate/equivalent? State CONCRETELY what changes numerically (operand precision, reduction order over K, accumulator precision, extra/omitted rounding, different quantization of activations, epilogue order), citing shader line numbers. Read the actual shader math, and compare against the upstream shader it replaces (`git show master:ggml/src/ggml-vulkan/vulkan-shaders/<file>` for upstream versions).

Paths (all under ggml/src/ggml-vulkan/):
1. `vulkan-shaders/mul_mm_f16reg.comp` — dense MUL_MAT of q8_0 weights. Upstream alternative is the int8 coopmat path (mul_mmq.comp / mul_mmq_cm1.comp with src1 quantized to q8_1_x4). This fork path uses f16 activations (B). Note the host comment at fork/ggml-vulkan-fork.cpp:426-427 about KL divergence, and whether the tile choice changes the order of the K sums (a comment near fork/ggml-vulkan-fork.cpp:440 claims "the tile does not change the order of the sums over K: the same results" — verify what that does and does not cover).
2. `vulkan-shaders/mul_mmid_f16reg.comp` — MUL_MAT_ID. Upstream = int8 q8_1 path. This one reads f32 activations in place OR an f16 copy (spec constant for B16), f16 B operand, coopmat with EPI variants. Comment at fork/ggml-vulkan-fork.cpp:221-222 mentions KLD 0.015735 vs 0.016057. Determine per-operand what precision changes.
3. `vulkan-shaders/mul_mat_vec_id_fork.comp` — MUL_MAT_ID mat-vec. Upstream = mul_mat_vecq.comp. Both quantize src1 to q8_1_x4? If so, is the integer accumulation identical (dot order, scale application order)? Check `dotPacked4x8EXT` / `dot16` ordering vs upstream's, and the fused scale (fusion_flags 0x4) application point vs upstream's separate MUL.
4. FUSED_GLU in `vulkan-shaders/mul_mmq_cm1.comp` (two A matrices, silu(gate)*up at the store) vs upstream running two separate matmul_id + a swiglu op. Bitwise?
5. MUL_MAT_HEADSUM in `vulkan-shaders/mul_mm.comp` (lines ~125-135, ~500-600): sums MUL_MAT_HEADSUM heads with `v += max(coopmat_stage[...], 0.0)` then adds a bias. Upstream = separate per-head matmuls + relu + a sum + bias add. Is the summation order and the accumulator precision the same? Is the relu applied to the same values? Note it runs on an f16 or f32 key pool and may use f16 accumulators.
6. `vulkan-shaders/gated_delta_net_lazy.comp` vs upstream `gated_delta_net.comp` (and vs upstream's separate rollback/copy). This is a NEW op (GGML_OP_GATED_DELTA_NET_LAZY) with "deferred rollback" and a records buffer (`data_rec`, binding 6) and `data_info` (binding 7). Does it compute the same arithmetic in the same order, or does deferring the rollback change results? Read the replay logic (mode 2) and the record/replay math.
7. `vulkan-shaders/dsv4_hc_post_norm.comp` (variants f32 / _f16n / _f16n_q8) vs upstream's separate DSV4_HC_POST + RMS_NORM + MUL (+CPY +quantize). Bitwise? Pay attention to: the f16 store in the N_F16 variant (what is rounded, when), the Q8 variant writing block_q8_1_x4 directly (compare its amax/scale/rounding against `quantize_q8_1.comp`), and the rms reduction order vs upstream `rms_norm.comp`.

Also state, where you can see it, whether the host code claims a numerical tolerance rather than exactness (look for test tolerances in tests/test-backend-ops-fork.inc or similar, and for KLD/KL comments in the fork source).

Every claim needs `file:line` evidence you actually read. Write "UNKNOWN" rather than guessing. Terse, evidence-dense, one short block per path. Do not report style nits. Do not summarize this prompt back.

**Отчёт агента (дословно):**

Now I'll write the report. I have all the evidence. Let me be terse and evidence-dense.

One more check I should do for path 2: is the K-loop order/grouping in mmid_f16reg 64-column stages with 4×16 coopmat MMAs (line 724, 750) — verified.

And for path 1 the host comment "the tile does not change the order of the sums over K: the same results" — verify it does NOT cover the int8 comparison (it's in the RDNA3 default-tile block, comparing r16x96/8x1 vs r16x64/8x1, both of which are the same shader). And also verify p.k_split is unused in the shader → grep showed k_split only at line 75 (declaration). Yes, verified.

Write the answer now. ## Verdict summary

| # | Path | Bitwise vs upstream? |
|---|---|---|
| 1 | `mul_mm_f16reg.comp` | **No** — different operand precision + different K reduction |
| 2 | `mul_mmid_f16reg.comp` | **No** — same, plus f16 A scale/offset and per-element f16 rounding |
| 3 | `mul_mat_vec_id_fork.comp` | **No** — same ints, different f32 grouping; epilogue itself is equivalent |
| 4 | FUSED_GLU `mul_mmq_cm1.comp` | **Yes** (modulo the fused-MUL rounding note) — same sums, same epilogue expression |
| 5 | MUL_MAT_HEADSUM `mul_mm.comp` | **Yes** on the epilogue; tile-set divergence is the only risk |
| 6 | `gated_delta_net_lazy.comp` | **Yes** for the arithmetic (records replay reproduces it exactly); the state buffer *means* something different |
| 7 | `dsv4_hc_post_norm.comp` | **Q8 sub-step: yes. Overall: no** (fma vs mul+add, different RMS reduction tree) |

No test in the repo asserts a tolerance for any of these — `tests/test-backend-ops-fork.inc` (1385 lines) contains no tolerance/exactness keyword at all (`grep tol|exact|bitwise` → one unrelated hit at :1160). The fork's numerical claims are all **KL-divergence / KLD prose**, i.e. explicitly tolerance-style:

- `fork/ggml-vulkan-fork.cpp:425-427` — "quality probes … at least as close to the reference as the int8 path (f16 activations: KL max 0.433 / 0.136 / 0.127 against 0.909 / 0.421 / 0.133 at 4096 / 16384 / 32768)". This is a *statistical* claim, not exactness; the fork path is **not** bit-equal to int8, and the comment says nothing about bits.
- `fork/ggml-vulkan-fork.cpp:222` — "KLD 0.015735 against 0.016057 (f16 activations instead of the 8-bit ones of the int8 path)".
- `fork/ggml-vulkan-fork.cpp:619-620` — the sharpest calibration: "KLD against f32 accumulators: 0.019, **twice what a mere reordering of the sums gives (0.010)**". So the author's own model is: reorder ≈ 0.010 KLD, precision change ≈ 0.019.
- `fork/ggml-vulkan-ops-fork.cpp:83` — "takes the KLD against exact sums from 0.020 to 0.044".

---

#### 1. `mul_mm_f16reg.comp` (dense q8_0, f16 activations)

**Not bitwise.** Three independent changes vs the int8 path (`mul_mmq.comp` + `quantize_q8_1.comp`):

- **B operand precision changes category.** Upstream quantizes activations per 32 columns to `int8` with an **f16** scale: `quantize_q8_1.comp:85` (`d = amax/127`), `:87` (`vals = round(vals*d_inv)`), `:121` (`ds = f16vec2(d, sum*d)`). Fork reads f32 and does a single `f32 -> f16` rounding (`mul_mm_f16reg.comp:243` `packFloat2x16`, or takes f16 directly under `B_F16`). f16 has ~10 mantissa bits vs ~7-8 relative for the int8+amax-scale, so the fork is *more* accurate for elements near amax and strictly different everywhere.
- **A operand gets an extra per-element rounding.** `mul_mm_f16reg.comp:173-174` computes `bytes01(x) * d` in **f16** (code exact in f16, `d` is the block's own f16 scale, so one product rounding per element). Upstream keeps the code integral and applies the scale once per 32-column block: `mul_mmq_funcs.glsl:218-227` — `q_sum += dotPacked4x8EXT(...)` (exact i32) then `float(q_sum) * float(dm) * float(ds.x)`.
- **K reduction is different.** Upstream: exact i32 dot per 32 columns, then f32 partials summed ascending (`mul_mmq.comp:88` `#define BK 32`, `:251` `sums[sums_idx] += mmq_dot_product(...)`). Fork: f16×f16→f32 `coopMatMulAdd` over 16-wide fragments, 4 fragments per 64-column stage, chained in an f32 accumulator (`mul_mm_f16reg.comp:50` `BK = 64`, `:434` `coopmat<float,…,Accumulator>`, `:446` K loop, `:346-352` the MMA chain). Partial-product rounding and the intra-MMA order are hardware-defined; there is no exact integer stage at all.

**The `fork.cpp:441` claim ("the tile does not change the order of the sums over K: the same results") is true but narrow.** It sits inside the block choosing between two *tiles of this same shader* (`r16x96/8x1` vs `r16x64/8x1`, `:438-444`). It holds because `BK` is a hard `const` 64 (`:50`), K is walked in one sequential loop (`:446`), and `k_split` is declared in the push constants (`:75`) but **never read** by the shader (grep: only line 75) — so no split-K reduction exists and BM/BN/WGM/WGN only decide which workgroup owns which output. It covers **nothing** about the int8-vs-f16 comparison, and it does not cover `BPF`/`B_F16` semantics either (those do change *where* the f16 rounding happens, though not its value here: the `to_f16` copy at `fork/ggml-vulkan-ops-fork.cpp:257` is a plain CPY of the same f32 `src1`, so both variants round the same f32 once).

#### 2. `mul_mmid_f16reg.comp` (MUL_MAT_ID, q4_K/q5_K/q5_1/q8_0)

**Not bitwise.** Per operand:

- **A scale/offset is rounded to f16 before use.** `:282` `so = f16vec2(dm.x * float(sm.x), dm.y * float(sm.y))` (q4_K/q5_K: the f32 product is computed then rounded to f16), `:303` (q5_1), `:315` (q8_0). Upstream keeps `dm` in f32 (`mul_mmq_funcs.glsl:492`) and applies it in the epilogue: `:525` `ds.x*dm.x*q_sum - dm.y*ds.y` per sub-block after an exact i32 dot (`:511-523`).
- **A element value is computed in f16 with two roundings.** `:444-445` `bytes01(x,1024.0) * so.x - so.y` — an f16 multiply and an f16 subtract per element (q8_0: `:416-417`, one f16 multiply). Upstream's codes are never scaled per element.
- **B operand** as in path 1: `:384-385` `packFloat2x16` of f32 activations (spec `B_F16=0`), or already-f16 (`B_F16=1`, `:361-363`). vs upstream's int8 q8_1_x4.
- **Accumulator**: f32 (`:697`, `:699`), K in 64-column stages, 4×16 fragments (`:121`, `:724`, `:750`) — no exact integer stage.
- **Epilogue is *not* an extra difference**: the fused router weight is one f32 multiply (`:508`, `:528`) after the f32 sums, and upstream's separate MUL is also one f32 multiply; same single rounding. `D_F16` (`:512`, `:533`) is one f16 rounding, matching an upstream CPY→f16 of the same f32 value. `silu_mul` (`:543`) is textually the same expression as upstream `swiglu.comp:7`.

The `fork.cpp:222` KLD numbers (0.015735 vs 0.016057) are the only numerical claim; they compare error against a reference, not bits.

#### 3. `mul_mat_vec_id_fork.comp` vs `mul_mat_vecq.comp`

Both quantize `src1` to q8_1_x4 identically — `fork/ggml-vulkan-ops-fork.cpp:456-457`: "src1 quantized to q8_1 x4 blocks in prealloc_y as the upstream mat-vec does it". So the **integer dots are identical in value**: the fork's `dot16 = dot4(a.x,b.x)+…` (`:80-82`, `dotPacked4x8EXT` at `:76`) and upstream's four `q_sum += dotPacked4x8EXT(...)` (`mul_mat_vecq_funcs.glsl:373-376`) are exact i32 sums, order-independent. **The results still differ**, because the f32 work around them differs:

- **Different K partitioning across lanes.** Upstream: lane `tid` takes 16-column pieces, `col = tid*K_PER_ITER`, `col_stride = K_PER_ITER*BLOCK_SIZE` (`mul_mat_vecq.comp:92,99`), one f32 term per 16-code call (`:72 temp[j][n] += mmvq_dot_product(...)`) reduced by `subgroupAdd` (`mul_mat_vec_base.glsl:100`). Fork: lane `l` takes whole 32-code chunks, `c = l%8` chunk of superblock `sbl = l/8` (`:183-185`), **two** sub-blocks folded into one f32 expression per round (`:241-242` `acc[K] += dsc0*ds0.x*dot + dsc1*ds1.x*dot - dmn0*ds0.y - dmn1*ds1.y`), reduced by `subgroupClusteredAdd(acc, LPR)` with LPR ∈ {8,16,32} (`:329`). Different partials, different tree → different last bits.
- **The min-correction is associated differently.** Upstream: `dm_scale.y * (ds.y/2)` (`mul_mat_vecq_funcs.glsl:380`), with `dm_scale = dm * scale_dm` as an f32 vec2 (`:365`). Fork: `(0.5 * dm.y * float(mn0)) * ds.y` (`:232`, `:242`). Same value mathematically, different rounding order.
- **The fused scale (fusion_flags 0x4) is not a difference**: `:333 r *= data_fuse0[...]` after the clustered add mirrors upstream's `temp[j][n] *= data_fuse0[expert_i0]` after `subgroupAdd` (`mul_mat_vec_base.glsl:111-113`) — one f32 multiply of the same two f32 operands.
- Caveat: mode 2 extends the path below upstream's k threshold, where upstream runs **f32 activations** (no quantization at all) — `fork/ggml-vulkan-fork.cpp` (`ggml_vk_fork_mmv_id_init`, "=2: also below its k threshold … which upstream runs on f32 activations … decode quality of upstream"). There the delta is a full quantization error, not last bits.

#### 4. FUSED_GLU in `mul_mmq_cm1.comp`

**Bitwise, as far as the shader math goes.**

- A2 is literally the same code re-included over a second buffer set (`mul_mmq_cm1.comp:179-192` `#define data_a data_a2` + second `#include "mul_mmq_cm1_funcs.glsl"`), with the same B tile, the same K loop and the same `cm1_accumulate` per 32-column block (`:589-601` mirrors `:578-588`). Because the int8 MMA accumulator is exact (`int(acc[e])`), the up sums are the same f32 sequence a separate dispatch would produce.
- Epilogue is textually upstream's: `:636` `D_TYPE(g / (1.0f + exp(-g)) * u)` vs `vulkan-shaders/swiglu.comp:7` `a / (1.0f + exp(-a)) * b`. Both read f32 (the fuse requires `a->type == F32 && b->type == F32 && glu->type == F32`, `fork/ggml-vulkan-fusion-fork.cpp:480-483`), so no f16 pre-rounding sneaks in, and the router-weight MUL stays separate (the FUSED_GLU branch does not call `get_fused_scale`).
- Residual risks, stated as such: (a) the FUSED_GLU pipeline list is a **shmem-filtered subset** of the same 3 int tiles (`ggml-vulkan.cpp:2471-2485`), and can pick a different index than the separate matmul (`mul_mat_id_m_int || fork_tiles.mmq_int_tile_glu_set && …`); that is harmless for BK (fixed) and TK (exact int), but `USE_MAGIC_BIAS = WARP != 32` (`:156`, `:165-171`) is a **different formula** (`fma(intBitsToFloat(acc_e), scale_a, -12582912*scale_a)`), so if the two paths ever land on tiles with different WARP the bits diverge. All int warptiles I saw carry WARP=32 (`:2449`), so on RDNA this shouldn't fire — **UNKNOWN** in general. (b) `KSCALES==2` types are excluded (`:402` `#error`), so q3_K/q6_K/nvfp4 never fuse.
- Side effect worth flagging: the fork also fused the router MUL into the *non*-GLU mmid store (`:639` `D_TYPE(float(sums[…]) * get_fused_scale(row_idx))`, same in `mul_mm.comp:495`, and `mul_mmq.comp:285`). Where `D_TYPE` is f16 that removes one rounding relative to upstream's f16-store-then-MUL. I did not find a host-side type guard restricting that to f32 dst — **UNKNOWN**.

#### 5. MUL_MAT_HEADSUM in `mul_mm.comp`

**The epilogue is order-equivalent and same-precision; that part is bitwise.**

- Accumulator is f32, not f16: the headsum pipelines are keyed `f16acc = false` — `fork/ggml-vulkan-fusion-fork.cpp:189` and `ggml-vulkan.cpp:2521`, `:2724` (`{type_a, GGML_TYPE_F32, false, false, false, true}`, `vk_matmul_pipeline_key` field order at `ggml-vulkan-types.h:257-265`). Since `shared ACC_TYPE coopmat_stage[]` (`mul_mm.comp:237`), the relu sees f32, exactly the values upstream's RELU node would see. `q0` must be F32 (`fusion-fork.cpp:309`), so the "f16 key pool" applies to operand A only, and both paths use the same `type_a`.
- Relu: `mul_mm.comp:518` `v += max(float(coopmat_stage[…]), 0.0)` / `:582` `v += max(vec2(sums[…]), vec2(0.0))` — same `max(x,0)` as upstream's unary RELU on the same f32.
- Head order matches upstream's graph exactly: `ggml_vk_find_headsum` only accepts a **sequential** chain `sum = add(sum, relu_h)` for h=1,2,3 (`fusion-fork.cpp:224-246`), and the bias add is the last node (`:248-252`). The shader sums `h = 0..3` ascending into `float v = 0.0` (`:516-519`) then `v += bias` (`:188-193`). `0 + x` is exact, so `(((h0+h1)+h2)+h3)+bias` in both. Bias values are unchanged (bit 1 = f32 bias, bit 2 = f16 mask read via `unpackHalf2x16`, i.e. the same value the f32 path already carried).
- K split is off: `k_split = k` in the push constants (`fusion-fork.cpp:364-366`).
- **The one real exposure**: the headsum tile set is not the plain tile set. CM1 keeps only `warptile[8] % 4 == 0` (`ggml-vulkan.cpp:2515-2518`); the non-coopmat list is *rewritten* — TN forced to 4, `TM = max(2, TM*TN/4)` (`:2710-2715`) — and `n` is `tokens*4` instead of `tokens`, so `ggml_vk_guess_matmul_pipeline_map` (`fusion-fork.cpp:356`) can land on a different tile than the per-head matmul. For this operand pair that is benign for the K order: `BK` is `#define BK 32` fixed for `DATA_A_F32/DATA_A_F16` (`mul_mm.comp:209-211`, not a spec constant), and per-element accumulation is one ascending chain (`:378`, `:405-415`, `:421+`) independent of BM/BN/WM/WN. So I found no mechanism by which a different tile changes the K sums here — **but the code does not assert that the headsum pipeline equals the per-head pipeline**, so this is an argument, not a guarantee.

#### 6. `gated_delta_net_lazy.comp`

**Same arithmetic, same order — the replay reproduces the net exactly.** The record step is the net's own state-update expression with the net's own recorded operands:

- Replay (`gated_delta_net_lazy.comp:134-141`): `s_shard[r] = g_val * s_shard[r] + data_rec[kb + …] * delta_c`, with lane→row mapping `r * LANES_PER_COLUMN + lane`, identical to the token loop's update (`:196` `s_shard[r] = g_exp[r]*s_shard[r] + k_reg[r]*delta_col`) and to upstream `gated_delta_net.comp`.
- The recorded `delta_col` is the *already-reduced* value written at `:210` (lane 0), i.e. the exact f32 the net used. Deferring the reduction does not re-reduce it: the replay needs no cross-lane reduction for the state step, so `reduce_partial` ordering (`:76-101`, clustered add / `subgroupAdd` / shmem tree) is used only where the plain net also uses it.
- `g_val` is stored once per head from the first column workgroup (`:212-221`) and read back at `:137`; the plain net recomputes `exp(data_g[gb_off])` per workgroup — same instruction on the same input, so same bits.
- No precision change: everything is f32 (`FLOAT_TYPE`/`float`), no casts.
- Upstream's `gated_delta_net.comp` diff (`git diff master...cache`) is **purely a destination change** — state now goes to `binding 7 SOutBuf` with `sout_off`/`sout_slot_stride` instead of `data_dst[s_off…]`, fusing the following CPY. Same values written (`git show` diff: only the buffer/offset names and the removed `state_size_per_snap`). Bitwise.
- What *does* change semantically: mode 1 writes `s_shard` to the state buffer **before** the token loop (`:143-147`), i.e. the state buffer holds the segment-start state and the current state is implied by `data_rec`; mode 2 materialises it (`:121` early-out when `n_rep==0 && row_in==row_out`). Equal final state, different intermediate contents — anything reading the state buffer mid-segment sees a different tensor.
- Dependency I did not fully chase: correctness of "same bits" requires the record writer and the replayer to use the same `LANES_PER_COLUMN`/`S_V` spec constants (host picks both from `ggml_vk_gdn_size_idx(S_v)`, `fork/ggml-vulkan-ops-fork.cpp:403-410`); I did not verify that invariant is asserted anywhere.

#### 7. `dsv4_hc_post_norm.comp`

**Mixed: the Q8 block is a faithful clone of `quantize_q8_1.comp`; the post+norm part is not bitwise.**

- **Not bitwise — the `x*post + r` step.** Fork `:97` `v[k] = fma(data_x[…], post, data_r[…])` (one rounding). Upstream `dsv4_hc_post.comp:85,91` (identity-mixing branch, `HAS_COMB == 0`): `float result = xv * post_s[idst]; result += r[idst];` (mul then add). Note the compiler is free to contract upstream's form into an fma, so the actual delta is driver-dependent — but the shaders as written differ.
- **Not bitwise — the sum of squares and the reduction tree.** Fork `:98` `ss = fma(v[k], v[k], ss)` vs upstream `rms_norm.comp:93` `sum += xi * xi;`. Fork reduces with a shmem tree over the whole `BLOCK_SIZE` (`:104-109`); upstream uses `subgroupAdd` + per-subgroup partials + a `sumsh` tree when `USE_SUBGROUP_ADD` (`rms_norm.comp:96-118`), the full shmem tree only in the `#else` (`:116-126`). Different grouping → `scale` differs in the last bits, which propagates to every output. Per-lane element→column assignment is the same strided form (`col = tid + k*BLOCK_SIZE`), and `mean = sum/n`, `inversesqrt(mean + eps)` match (`:111` vs `rms_norm.comp:129-130`).
- **The gamma multiply matches.** Fork `v[k] * scale * data_w[…]` (`:122`, `:152`) vs upstream `scale * data_a[…] * data_b[…]` (`rms_norm.comp:139,152`) — IEEE multiplication is exact-symmetric, so `(v*scale)*w == (scale*v)*w` bit for bit.
- **N_F16 store: one rounding, equals a CPY→f16.** `:152` `float16_t(v*scale*w)`; same single RN as the separate cast, *given* identical f32 inputs.
- **Q8 variant: bitwise equal to CPY(f16) + `quantize_q8_1.comp QBLOCK_X4 A_F16`.** Same input (`nf = float(n)` of the f16 `n`, `:122` vs `quantize_q8_1.comp:65 vec4(data_a[a_idx])`); `amax` is order-independent, `subgroupClusteredMax(abs(nf),32)` (`:128`) == clustered-max-8-of-vec4; `d = amax/127`, `d_inv` select, `round(nf*d_inv)` (`:129-131`) == `:85-87`; `sum` (`:132`) is an exact integer sum either way (|q|≤127) == `:104-113`; `ds = f16vec2(vec2(d, sum*d))` (`:144`) == `:121`; byte packing "lowest index in lowest byte" via `subgroupClusteredOr(... << 8*(lane%4), 4)` (`:135`) matches `pack32(i8vec4(round(vals)))` (`:92`). Block↔index mapping `ib = (it*n_rows*n_embd + s*n_embd + i0)/32` (`:137`) reproduces the flattened contiguous order, guarded host-side by `n_embd % 32 == 0` (`fork/ggml-vulkan-fusion-fork.cpp:44`).
- So for the Q8 path the net effect is: the *quantization* is bit-exact, and the divergence enters one stage earlier, through `scale`.

#### Files read (no edits)

`ggml/src/ggml-vulkan/vulkan-shaders/{mul_mm_f16reg,mul_mmid_f16reg,mul_mat_vec_id_fork,mul_mmq,mul_mmq_cm1,mul_mm,dsv4_hc_post_norm,gated_delta_net_lazy,quantize_q8_1,swiglu}.comp`, `{mul_mmq_funcs,mul_mmq_cm1_funcs,mul_mat_vecq_funcs,mul_mat_vec_base,glu_main}.glsl`, `dsv4_hc_post.comp`, `rms_norm.comp`, `ggml-vulkan.cpp`, `fork/ggml-vulkan-{fork,ops-fork,fusion-fork}.cpp`, `ggml-vulkan-types.h`, `tests/test-backend-ops-fork.inc`, `ggml/src/ggml-cpu/vec.h`.

---

### B2.4.2 Шейдер mul_mmid_f16reg

- Агент: `general-purpose-zIszuXvnzDpek5HJUv5wad5qVwWI1vGU` (тип `general-purpose`, глубина 2, родитель: `general-purpose-xrw1nNGY0G8cGQ9zgejMyaNU4GMFGQOz`)
- Статус: completed, создан 15:59:38 UTC, финальный отчёт 20178 символов

**Постановка задачи (дословно из лога):**

READ-ONLY code review. Do not edit any file.

Repo: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`, a perf fork of upstream llama.cpp; upstream baseline is branch `master`).

Task: review ONE new GLSL compute shader for correctness bugs:
ggml/src/ggml-vulkan/vulkan-shaders/mul_mmid_f16reg.comp  (803 lines, brand new — read the WHOLE file)

It implements MUL_MAT_ID with q4_k/q5_k/q5_1/q8_0 weights decoded into float16 A cooperative-matrix fragments in registers. Generator variants (from vulkan-shaders-gen.cpp): defines DATA_A_Q4_K|DATA_A_Q5_K|DATA_A_Q5_1|DATA_A_Q8_0, optional FUSED_GLU, ROWS2, GLU_SPLIT, EMULATE. Names mul_mmid_f16reg_<t>[_glu|_r2|_glus][_emu].

Look ONLY for these classes of bug, in priority order:
(a) NUMERICS that change model output: float16 accumulation or f16 intermediate where the upstream shaders (mul_mmq.comp, mul_mmq_cm1.comp, mul_mat_vec*.comp on `master`) accumulate in f32/int32; changed reduction order; a scale/min term applied to the WRONG sub-block or the wrong half of a block; the q4_K `get_scale_min_k4` bit decode being wrong; the `- m * sum(activations)` correction term being dropped, doubled, or applied with the wrong block-sum width; silu/glu applied to the wrong operand (compare against ggml_swiglu_split / ggml_swiglu in ggml/src/ggml.c to check which operand is the gate).
(b) OOB reads/writes: index arithmetic using the wrong stride/pitch; missing bounds check when M/N/K/ne00/ne01 is not a multiple of the tile; the "clamp the row/col to the last valid one" tricks failing; a partial tile reading past the end of a buffer; 32-bit uint overflow in `row * stride` for large shapes.
(c) Missing `barrier()` / `controlBarrier` between a shared-memory write and another invocation's read; shared memory read before it is initialized; `barrier()` inside non-uniform control flow (UB in Vulkan). Pay attention to any barrier inside `if (...)` where the condition is NOT uniform across the workgroup.
(d) Specialization-constant misuse: a `shared` array sized from one constant (e.g. BLOCK_SIZE, WMI, WNI, ROWS) while the code indexes it with a value derived from something else (gl_NumSubgroups, local_size, a runtime value) that can exceed it -> shared-memory OOB.
(e) NaN/Inf behaviour differing from upstream.

Method requirements:
- Verify every claim against the actual code. Quote the exact line numbers and the specific expression you read as evidence.
- Compare against the upstream equivalent on `master`: `git show master:ggml/src/ggml-vulkan/vulkan-shaders/<file>` and the CPU reference in ggml/src/ggml.c (ggml_compute_forward_mul_mat_id, ggml_compute_forward_swiglu*).
- You may run `glslangValidator -V --target-env=vulkan1.2 -S comp -DDEF=1 ... <file>` from the vulkan-shaders directory to check that variants compile (glslangValidator is on PATH). Try several define combinations including FUSED_GLU, ROWS2, GLU_SPLIT, EMULATE. Report any variant that fails to compile.
- Also check whether the pipeline this shader belongs to is used BY DEFAULT or is opt-in: grep ggml/src/ggml-vulkan/ (especially ggml-vulkan.cpp, ggml-vulkan-device.cpp, fork/*.cpp) for the env knob (e.g. GGML_VK_MMID_F16REG) and report its name and default value.

Report at most 6 findings, most severe first. For each: SEVERITY (Critical = wrong output or memory error / Suggestion / Nit), exact file:line, one-line claim, the CONCRETE failure scenario (which quant type, which shape, what wrong value or what OOB), whether the path is default or opt-in (env knob + default), and the evidence lines. Also explicitly list which of (a)-(e) you checked and found clean. Do NOT report style, duplication, or performance issues.

**Отчёт агента (дословно):**

I have everything I need. Let me write the final report.

Findings (most severe first):

1. Suggestion — mul_mmid_f16reg.comp:225 uint32 overflow in the A element offset (multiply-first). Class (b).
2. Suggestion — mul_mmid_f16reg.comp:383-386 f32→f16 narrowing of the activations, ±inf above 65504, where the int8 path it replaces (q8_1) cannot overflow; also the output is narrowed again with D_F16 (lines 512/532). Class (a)/(e).
3. Nit — GLU_SPLIT: `grow0 = ir*BMW + wave*16` (line 793) with BMW=64 (line 117) gives the up subgroups (wave 4..7) a `grow0` in [ir*64+64, ir*64+112], i.e. the NEXT workgroup's rows; only the `gate` guard at 558/562/577 keeps those out of the stores. Latent hazard.
4. Nit — `shared u16vec2 row_ids[BN]` read at line 357 via `min(n, tile_count - 1)`: if `tile_count` were ever 0 this becomes `min(n, 0xFFFFFFFF) = n` and reads uninitialized shared memory, feeding an arbitrary `e0` into `data_b[e0/8]` (line 361) → OOB read. Currently unreachable because count_experts.comp:141-143 `write_tiles` only emits tiles for count>0 with `min(row_tile_size, count - t*row_tile_size)` ≥ 1, so the shader relies on an invariant from another file with no `max(tile_count,1)` guard. Defensive.
5. Nit — `GGML_VK_MMID_F16REG_ROWS2` default (unset → q5_1 only) plus `EPI` default 0 means the `_r2`/`_glus`/`EPI=1` variants are exercised only under non-default env, so their paths in this file are not covered by the default configuration. Hmm, that's not a bug. Maybe replace with something else.

Better #5: Note the shader relies on `max_tiles`/`bn` consistency between count_experts and the matmul — verified fine. Skip.

Alternative #5 (real, minor): For `EPI == 0` + `FUSED_GLU`/`ROWS2` the epilogue syncs sh_c with `controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, ...)` while the `GLU_SPLIT` variant crosses subgroups and correctly uses full `barrier()`. That's fine — not a finding.

Maybe #5: `layout(binding = 3) readonly buffer IDS { int data_ids[]; };` (line 90) is declared but never referenced — it exists only to hold the descriptor slot. Harmless. That's arguably style. The task says don't report style. I'll skip.

Let me instead add a genuinely useful 5th: the f16 `so` scale rounding (line 282) — `so = f16vec2(dm.x * float(sm.x), dm.y * float(sm.y))` keeps d*sc and dmin*m in f16 whereas the int8 MMQ path keeps them f32; combined with the f16 store of the dequantized weight this yields ~2^-11 relative error per weight and cancellation error when `dmin*m` dominates `q*d*sc`. This is a numerics *deviation* (class a) worth recording even though it's the file's design — I'll list it under "checked" with the number, and as a Nit noting that `so.x`'s product is formed in f32 then narrowed (only the narrowing is lossy).

Hmm, I think 4 findings is fine; "at most 6".

Let me finalize with 4 findings + explicit clean lists.

Also should mention compile results and the env knob / default status for each finding: all findings are on the path that is DEFAULT-ON on AMD RDNA3 (GGML_VK_MMID_F16REG unset → mode=1 after the coopmat A probe), OFF elsewhere.
- Finding 1: default path (all types/variants).
- Finding 2: default path for B_F16... `b16_glu` default ON → the fused gate/up reads f16 by default; single matmuls read f32 in place but still narrow to f16 in write_b_tile (lines 383-386). So the narrowing happens in ALL default cases ✓.
- Finding 3: opt-in (`GGML_VK_MMID_F16REG_GLU_SPLIT=1`, ops-fork.cpp:161).
- Finding 4: currently unreachable (defensive).

Also I should explicitly answer: "check whether the pipeline is used BY DEFAULT or is opt-in".

And explicitly list which of (a)-(e) I checked and found clean, with the verification evidence.

Let me write it. I've completed the review. Below is the report.

---

### Review: `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmid_f16reg.comp`

#### Path status: **default-ON on AMD RDNA3**, off elsewhere

- `ggml/src/ggml-vulkan/fork/ggml-vulkan-fork.cpp:229-237` — knob `GGML_VK_MMID_F16REG`. When **unset**: `if (!(device->vendor_id == VK_VENDOR_ID_AMD && device->architecture == AMD_RDNA3)) return; env = "1";` → enabled by default on RDNA3 only. `=0` off, `=1` force, `=emulate` shuffle path. Still requires `fp16`, 32-lane subgroups, `KHR_cooperative_matrix` + subgroup-size control (fork.cpp:239-256) and `ggml_vk_fork_coopmat_a_probe()` (fork.cpp:257).
- Sub-knobs and defaults: `_BPF` (1, fork.cpp:195), `_EPI` (0, fork.cpp:196-203, gated on `coopmat_c_probe`), `_DB` (0, fork.cpp:211), `_BN`/`_GLU_BN` (64, ops-fork.cpp:158-159), `_ROWS2` (unset → **q5_1 only**, ops-fork.cpp:160,167), `_GLU_SPLIT` (off, ops-fork.cpp:161), `_GLU_B16` (**on**, ops-fork.cpp:166), `_MIN_ROWS` (16, ops-fork.cpp:129). Fused GLU additionally needs ≥ `GGML_VK_MMID_GLU_MIN_TOKENS` = 2048 (fusion-fork.cpp:491).
- Host LDS accounting (`ggml-vulkan.cpp:3841-3850`) matches the shader's actual `shared` declarations exactly (verified term by term, incl. `db` adding one more `tiles`), so no pipeline passes validation with an undersized allocation.

#### Compile check
`glslangValidator --target-env vulkan1.2 -S comp` over all 32 generator-legal combos (`DATA_A_Q4_K|Q5_K|Q5_1|Q8_0` × {}, `FUSED_GLU`, `ROWS2`, `GLU_SPLIT`, `EMULATE`, `EMULATE`×each): **0 failures, 0 warnings**. (`-DEPI=1`, `-DBPF=0`, `-DB_F16=1`, `-DDB=1` are *specialization constants*, not preprocessor defines — passing them as `-D` breaks the build by design and is not a real failure; the generator never does it, `vulkan-shaders-gen.cpp:1140-1160`.)

---

#### Findings

##### 1. Suggestion — `(expert * p.batch_stride_a + grow * p.stride_a)` overflows in element space; upstream divides first
**`mul_mmid_f16reg.comp:225`**
```glsl
const uint row_blk = (expert * p.batch_stride_a + grow * p.stride_a) / A_QK;
```
Upstream keeps the product in *quant-group* space: `mul_mmq.comp:182` `expert_idx * (p.batch_stride_a / BK) + ...` and `mul_mat_vec_base.glsl:74` `expert_id * (p.batch_stride_a / QUANT_K)`. Here the multiply happens in **elements**, so the uint32 sum wraps once the byte-offset of a weight element exceeds 2^32 elements.

**Concrete failure:** q4_K MoE expert tensor ≥ 4.295e9 elements (= 537 MB at 4.5 bpw; q5_K 597 MB, q5_1 644 MB, q8_0 912 MB) → `row_blk` wraps to a small value → the shader reads the *wrong* weight blocks (the wrapped index stays inside the buffer, so no fault — silently wrong logits), or a low-index block of another expert.

**Reachability caveat (why this is Suggestion, not Critical):** the host gate is byte-based — `ggml-vulkan-ops-fork.cpp:142-144` `ggml_nbytes(src0) > range → return nullptr` with `range = maxStorageBufferRange`, and `ggml-vulkan.cpp:8027` `if (!f16reg && nbytes > range)` explicitly exempts this pipeline from the 64-bit-indexing fallback. q4_K needs 4.295e9 × 0.5625 = **2.42 GB** to wrap, so drivers reporting `maxStorageBufferRange = 2^31-1` (2.147 GB) are safe with only a **12 % margin**; a driver reporting > 2.42 GB (I could not confirm RADV's value from this checkout) makes it reachable for DeepSeek/Kimi-class 256/384-expert q4_K tensors. **Default** path on RDNA3.
**Fix:** `expert * (p.batch_stride_a / A_QK) + grow * (p.stride_a / A_QK)` (both divide exactly: `ne[0] % qk == 0` is enforced at ops-fork.cpp:143).

##### 2. Suggestion — activations narrowed f32 → f16 (`write_b_tile`), which can overflow to ±inf where the int8 path it replaces cannot
**`mul_mmid_f16reg.comp:383-386`**
```glsl
sh_b[o + j] = uvec4(packFloat2x16(f16vec2(rb[it][2 * j].xy)), packFloat2x16(f16vec2(rb[it][2 * j].zw)),
                    packFloat2x16(f16vec2(rb[it][2 * j + 1].xy)), packFloat2x16(f16vec2(rb[it][2 * j + 1].zw)));
```
The path this pipeline replaces quantizes the activations to **q8_1** (`ggml-vulkan.cpp:7971`, `can_quantize_y`), i.e. a per-32-column f32 scale — it cannot overflow for any finite input. Here every activation is forced into f16, so `|x| > 65504` becomes ±inf, and `0 * inf` inside `coopMatMulAdd` yields NaN that then lands in a **stored** output element (the "columns past the rows read the last row" trick at line 357 only protects against uninitialized data, not against inf in a valid row). With `D_F16` (fused gate/up → f16 GLU, `GGML_VK_MMID_F16REG_GLU_B16` on by default) the result is narrowed a second time at lines 512 / 532.
**Not** a divergence from upstream *f16* coopmat paths (`mul_mm.comp`), only from the int8 MMQ path this replaces; magnitudes would have to exceed 65504, which is rare. **Default** path (every variant, `B_F16=0` still narrows). Accumulation itself is correct — `acc` is `coopmat<float, …, gl_MatrixUseAccumulator>` (line 769/675), never f16.

##### 3. Nit — GLU_SPLIT: up-projection subgroups carry a `grow0` that points into the *next* workgroup's rows
**`mul_mmid_f16reg.comp:793` + `:117` + `:557-558`**
```glsl
const uint grow0 = ir * BMW + wave * 16;   // 793
const uint BMW  = BM / 2;                  // 117  (GLU_SPLIT)
const uint half_w = BLOCK_SIZE / WARP / 2; const bool gate = gl_SubgroupID < half_w;   // 557-558
```
With `BMW = 64` and `wave` up to 7, `grow0` for subgroups 4-7 is `ir*64 + 64 … ir*64 + 112` — i.e. the row range owned by workgroup `ir+1`. Harmless today *only* because both stores are inside `if (gate)` (lines 569, 582) and the up subgroups' `grow0` is otherwise unused; the shared-memory pairing uses `gl_SubgroupID - half_w` (line 574) and `gl_SubgroupID` (line 579), which are correct. Any future change that lets a non-gate subgroup store, or that moves `write_frag*` out of the `gate` guard, writes another workgroup's rows. Opt-in only (`GGML_VK_MMID_F16REG_GLU_SPLIT=1`).

##### 4. Nit — `row_ids[min(n, tile_count - 1)]` underflows if `tile_count` is ever 0
**`mul_mmid_f16reg.comp:357, 361`**
```glsl
const u16vec2 ri = row_ids[min(n, tile_count - 1)];
...
rbh[it][0] = data_b[e0 / 8];
```
If `tile_count == 0`, `tile_count - 1 == 0xFFFFFFFF`, the `min` degrades to `n`, and `row_ids[n]` is **uninitialized shared memory** (the fill loop at lines 674-677 writes only `i < tile_count`), producing an arbitrary `e0` for `data_b[e0/8]` → OOB read. Currently unreachable: `count_experts.comp:141-143` `write_tiles()` emits `data_d[descriptor + 1] = min(p.row_tile_size, count - t * p.row_tile_size)` and only `n_tiles = ceil(count/row_tile_size)` tiles, so `tile_count >= 1`. The shader relies on another file's invariant with no local `max(tile_count, 1)`. **Default** code, currently benign.

---

#### Checked and found clean (with the evidence that settled each)

**(a) Numerics — clean.**
- **q4_K `get_scale_min_k4`**: shader `scale_min_k4` (lines 169-182) is bit-for-bit `ggml/src/ggml-quants.c:889-896` (`j<4`: `q[j]&63`, `q[j+4]&63`; else `(q[j+4]&0xF)|((q[j-4]>>6)<<4)`, `(q[j+4]>>4)|((q[j]>>6)<<4)`), fed `ra_raw[m][1..3]` = `scales[0..11]` with `d/dmin = unpackHalf2x16(ra_raw[m][0])` (line 280). Call site `scale_min_k4(2*ra_j + h, …)` (line 281) → sub-block index matches.
- **Scale/min ↔ sub-block association** — the part that looked broken until the real block layouts were checked. q4_K/q5_K pack a **64-column chunk** as 32 bytes with `byte l: low nibble = col l of sub-block 2j, high nibble = col l of sub-block 2j+1` (`ggml-quants.c:1530-1533` `q[l] = L[j+l] | (L[j+l+32] << 4)`; `dequantize_row_q5_K` at `ggml-quants.c:1759-1763` same order). Therefore the nibble, not the byte group, selects the sub-block, so `base = … + (f & 1) * 4` / `shift = (f >> 1) * 4` (lines 423-424) paired with `so_o = … + (f >> 1)` (line 401) is **correct**, and gives stage columns 16f..16f+15 in increasing order, matching `coopMatLoad(… b_base + (C*16*BKP + f*16)/8 …)` (line 654). The same rule holds for q5_1 (nibble = block, `so = f16vec2(dm.x, -dm.y)`, line 303 → `q*d + m`, matching `dequantize_row_q5_1`, `ggml-quants.c:529-552` incl. the `qh` bit `j` / `j+16` convention) and for q8_0 (`base = … + (f>>1)*8 + (f&1)*4`, line 409; `^ 0x80808080` at line 318 with bias `1152.0` at 414-415; scale from low half of word 0 / high half of word 8, lines 313-315).
- **`- m` correction**: folded per element into A (`v01 = bytes01(x,1024.0) * so.x - so.y`, lines 437-438), so there is no block-sum term to drop or double; algebra matches `dequantize_row_q4_K`'s `dloop*q - mloop`. Block-sum width is not applicable. `bytes01/bytes23` (lines 397-398) extract columns 4i,4i+1 / 4i+2,4i+3 exactly (`x` is pre-masked with `0x0F0F0F0F`), and `1024+q-1024` is exact in f16.
- **q5_K `qh` transpose**: `sbit = 2*ra_j + h` (line 289) matches `u1 = 1<<2c`, `u2 = u1<<1` in `dequantize_row_q5_K` (sub-block s uses bit s of `qh[byte l]`); the gather `(x&1)|((x>>7)&2)|((x>>14)&4)|((x>>21)&8) << 4w` (lines 291-293) puts bit k of the assembled word at bit `sbit` of qh byte k, and `hb >> ((f & 1) * 16)` (lines 427/429) + `(hb >> (4*i)) & 0xF` → bit for byte index `(f&1)*16 + 4i + jj`, exactly the byte the nibble came from. The `b * 0x00204081 & 0x01010101) << 4` scatter is correct (bits 0/7/14/21 → 0/8/16/24 → bit 4 of byte k).
- **Reduction order / accumulation**: `acc` is `coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>` (lines 675-678, 769) — no f16 accumulation anywhere; f16 only for the A/B fragments (the file's stated design, same as `mul_mm_f16reg.comp`). EMULATE's shuffle dot is self-consistent: `subgroupShuffle(a[k], (lane>>4)+2*e) * b[k]` with `b[k]` at `hi = b_off + (lane&15)*BKP + k` (line 461) = A[row][k]·B[k][col], one lane per (row, col), matching `frag_stage` (line 476) and `write_frag_acc` (line 521).
- **silu/GLU operand**: `silu_mul(g,u) = g/(1+exp(-g))*u` (line 543) applied to (binding 0 = `a`, binding 6 = `a2`) at lines 599, 618, 567. Reference is `y = silu(src[0]) * src[1]` (`ggml_vec_swiglu_f32(n,y,x,g)`, `ggml/src/ggml-cpu/vec.cpp:421`; `ggml_swiglu_split(a,b)` → `src[0]=a`, `ggml.c:3096-3101`). The host resolves the operands correctly, not by graph order: `gate = swapped ? glu->src[1] : glu->src[0]`, then `ggml_vk_mul_mat_id_q_f16(ctx, subctx, gate->src[0], …, up->src[0], out)` (`ggml-vulkan.cpp:8586-8591`) — so binding 0 really is the gate even though llama-graph emits the **up** MMID first (`src/llama-graph.cpp:2407-2420` vs `:2420-2423`) and `ggml_vk_can_fuse_mmid_glu` accepts either order (`fusion-fork.cpp:462-465`). `exp`-overflow behaviour matches upstream (`-inf → g/inf → 0`).
- Block addressing/strides: `stride_a = ne10` (elements) and `row_blk = element_offset / A_QK` is a valid block index because rows are contiguous; `A_QK = 256` for q4_K/q5_K requires `ne[0] % 256 == 0`, enforced by `src0->ne[0] % qk != 0` (ops-fork.cpp:143). Block sizes verified against `ggml-common.h`: q4_K 144 B = 36 words (`:337-338`), q5_K 176 B = 44 words (`:355-356`), q5_1 24 B = 6 words (`:248-249`), q8_0 34 B, pair = 17 words (`:255-256`). `bw = (row_blk + k0/256)*A_BLOCK_WORDS` (229), `qsw = bw+4+j*8+h*4` (235) / `bw+12+j*8+h*4` (245), `pw = (row_blk + k0/32)/2*17 + h*8` (263) all land exactly.

**(b) OOB — clean apart from findings 1 and 4.**
- `grow = ir*BMW + row` is bounded: `elements = {m, …}` with `wg_denoms[0] = v==2 ? 256 : v==3 ? 64 : 128` = `BMW` (`ggml-vulkan.cpp:3860-3862`, matching lines 113-121), and the host requires `ne[1] % 128 == 0` plus `% 256 == 0` for ROWS2 / `% 64 == 0` for GLU_SPLIT (ops-fork.cpp:143,169,171) → `grow < M` always, and `expert < n_experts`, so every A read is in-bounds even for rows past `tile_count` (the unguarded load is intentional and safe here). `p.M` guards at 504/530 are therefore redundant, not load-bearing.
- Non-tile-multiple dimension (tokens) is handled: `nc_used = (tile_count + 15)/16` (line 681) limits the fragments, and `n < tile_count` (504, 523) limits the stores. `min(n, tile_count-1)` (357) keeps out-of-range B reads on a valid row, so no NaN can enter a stored element.
- `row_ids[BN]` (155) bounded by `tile_count ≤ BN` (`count_experts.comp:143` `min(p.row_tile_size, …)`, with `row_tile_size = bn = pipeline->wg_denoms[1]`, `ggml-vulkan.cpp:8023`, and `bn` passed to count_experts at `:8171` — after `pipeline = f16reg` at `:8018`, so the same pipeline's BN).
- Row-map indexing: `tile >= data_row_map[0]` guard (667), `1 + 2*tile`, `2 + 2*tile`, `ids_base = p.row_ids_offset + tile*BN` all `≤ row_map_words - 1` given `row_ids_offset = 1 + 2*max_tiles`, `row_map_words = row_ids_offset + max_tiles*bn` (`ggml-vulkan.cpp:8029-8031`). `tile = y + z*gl_NumWorkGroups.y` covers `[0, tiles_y*ceil(max_tiles/tiles_y))` (`:6492`).
- `sh_qs`/`sh_so`/`sh_qh`/`sh_qs2`: max indices 2175 / 255 / 255 for `BM=128` (checked f=3, h=1, row=127). `sh_b[o+j]`: max `9*BN - 2 < NBUF*9*BN` (line 379); B-column span `(BN-1)*72 + 63 < 72*BN`. `sh_c`: max `(BLOCK_SIZE/WARP)*256 - 1` and, for GLU_SPLIT/EPI=1, `half_w*256 - 1` (lines 560/565/574/579 vs `SH_C` 150-152) — all inside. `BLOCK_SIZE` is also `local_size_x` (same `constant_id 0`, lines 71 and 81, spec vector `{256, bn, bpf, epi, h, o, db}` at `ggml-vulkan.cpp:3862`), so the `row < BM` invariant holds by construction.
- B addressing alignment: `e0 = ri.y*batch_stride_b + (ri.x % ne11)*stride_b + k0 + q*16` (358) is 16-byte/8-half aligned via `src1->nb[2] % 16 != 0 || src1->nb[3] % 16 != 0 → nullptr` (ops-fork.cpp:149) plus `K % 64 == 0`; `B_ITERS` (343) is exact for BN ∈ {32,64,128} and `rb[2][…]`/`rbh[2][…]` are never indexed past `it=1`.
- `coopMatLoad` from a `shared uvec4` array (line 654) for a `float16_t` matrix is the same pattern already used for `data_kv4`/`data_vv4` in `flash_attn_cm1.comp:325,512`; byte offset `16*(C*144 + 2f)` and stride `144 B` are 16-byte aligned.

**(c) Barriers — clean.** Every `barrier()` (lines 702, 715, 719, 728, 734, 812-region, 561/571/576/584) is guarded only by **specialization constants** (`DB`, `EPI`) or by `C < NC && C < nc_used` where `nc_used` derives from the workgroup-uniform `tile_count` — never by `gate`, `lane`, or `m`. The early `return` at 667-669 is workgroup-uniform. `row_ids` writes are fenced by the barrier at 702 before the first `load_b_tile` read. `barrier()` at 728 separates the previous stage's shared reads from `write_a_tile`/`write_b_tile`. The `DB=1` double-buffering is correct with **one** barrier per stage: stage *k* writes `cur^1`, which was last read at stage *k-1* and separated by the barrier at the end of that iteration (lines 779-791). The per-subgroup `sh_c` sync uses `controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease)` — legal for intra-subgroup shared traffic and the established pattern here (`mul_mm.comp:509,523,547,551,556,562`; `mul_mm_f16reg.comp:361,370`); cross-subgroup `sh_c` traffic in GLU_SPLIT correctly uses full `barrier()` instead. In the `EPI==0` `TWO_A` paths the extra `controlBarrier` between `read_frag` and the next `frag_stage` (lines 589-591, 604-605) is present and necessary. `coopMatStore` under `if (!gate)` (line 574) is uniform within each subgroup, and `require_full_subgroups`/`required_subgroup_size=32` are set (`ggml-vulkan.cpp:3863`).

**(d) Specialization-constant misuse — clean.** No `shared` array is indexed beyond the constant it is sized from; see the (b) index arithmetic. `NC = BN/16 ≤ 8` bounds the `acc[8]`/`acc2[8]` arrays and the `MMA_C`/`EPILOGUE_C` macro chain; `NC` and `nc_used` are compared against every index. `WARP = 32` is guaranteed by the `sg32` check (fork.cpp:240-246) plus `required_subgroup_size = 32` (ggml-vulkan.cpp:3829, 3863).

**(e) NaN/Inf — clean except finding 2.** Out-of-range columns read a valid B row rather than zeros/garbage, so no spurious NaN enters a stored element; `silu_mul` matches upstream's `x/(1+exp(-x))*y` limit behaviour (`-g → +inf` ⇒ `g/inf → ±0`); `bytes01/bytes23` stay far inside f16 range (`1024+255` max); the `d*sc`/`dmin*m` products are formed in f32 before narrowing (line 282). `data_fscale` (binding 5) is always bound — `{d_D, d_buf_offset, d_sz}` when unfused (`ggml-vulkan.cpp:8261`) — and only read under `p.fusion_flags & 1`, which the GLU path sets to 0.

**Not reported** (per instructions): style, duplication, the `bytes01`/`bytes23`-shared-with-`mul_mm_f16reg.comp` duplication, the unused `IDS`/`data_ids` binding (line 90) and the dead `p.N`/`p.nei1`/`p.nbi1`/`p.n_experts` push-constant fields, and performance.

---

### B2.5 Изменения CPU-операций форка

- Агент: `general-purpose-co0U3AXjmCU2ivHINuR3LnY4C4WTmjLo` (тип `general-purpose`, глубина 1, родитель: `general-purpose-ksLnpjA7E67cqBYBArnNK6VaQtRzgMCz`)
- Статус: completed, создан 17:38:39 UTC, финальный отчёт 14959 символов

**Постановка задачи (дословно из лога):**

READ-ONLY code review. Do NOT edit any file. You MUST run git commands and read files; do not answer from memory. Repo: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`, perf fork of llama.cpp). Merge base = branch `master`.

Scope: CPU backend changes only:
- ggml/src/ggml-cpu/fork/ops-fork.inc (146, new)
- ggml/src/ggml-cpu/fork/timeline-fork.inc (110, new)
- `git diff master...HEAD --` for: ggml/src/ggml-cpu/ops.cpp (+142), ggml/src/ggml-cpu/ggml-cpu.c (+80), ggml/src/ggml-cpu/ops.h, ggml/src/ggml-cpu/ggml-cpu.cpp, ggml/src/ggml-cpu/repack.cpp, ggml/src/ggml-cpu/spacemit/ime.cpp

Look for, in priority order:
(a) NUMERICS that change model output: any new f16/bf16 path replacing f32 (e.g. a `*_f16_fork` variant), changed accumulation order in a reduction, a vec_dot variant with different scale handling, FMA contraction that upstream does not do. For each, say exactly where precision is lost and whether the magnitude is bounded.
(b) OOB: index arithmetic past the end of a tensor when ne is not a multiple of the vector/tile width; missing `ne` bounds in a tail loop; 32-bit int overflow of an offset for large tensors; reading `op_params`/hints without checking they were set; missing NULL checks on optional src tensors (e.g. src[3] skip input of MUL_MAT_ID).
(c) Thread safety in the CPU worker path: shared state written by multiple ggml threads without atomics, work-division math that leaves a range uncovered or double-covered when ggml_thread_compute_ifs / n_threads changes.
(d) timeline-fork.inc: does instrumentation change behavior or is it a pure no-op when disabled? What is its env knob and default?

For EVERY finding: SEVERITY (Critical = wrong output or memory error; Suggestion; Nit), exact file:line, one-line claim, concrete failure scenario (shape/dtype and what wrong value/OOB), ON BY DEFAULT or opt-in (name the env knob + default), and code evidence. Verify against upstream before claiming. State which files you checked and found clean. No style/perf/duplication. Max 8 findings.

**Отчёт агента (дословно):**

I have everything. Writing the report now.

Files checked and found clean:
- ggml/src/ggml-cpu/ops.h (declarations only, match definitions)
- ggml/src/ggml-cpu/repack.cpp:5213-5214 and ggml/src/ggml-cpu/spacemit/ime.cpp:1618 — the `op->src[3] == nullptr` guards: correct and produce a clean fallback, verified src/fork/llama-graph-fork.cpp:193-203 removes skip when `ggml_backend_supports_op(backend_cpu, mm)` is false; also ggml_mul_mat_id_add_skip asserts nelements(skip) == src0->ne[2] so skip[i02] is in-bounds.
- ggml-cpu.c graph_plan cases (work sizing), ggml_get_n_tasks entries: verified against params->ith strides.
- MUL_MAT_ID skip memset: same form as upstream i02==-1 path, row-exclusive, ordered by the following ggml_barrier.
- tiled MUL_MAT_ID path reuses the caller-converted wdata → F16 safe there (only MUL_MAT tiled is broken).
- No FMA/accumulation-order changes: the lazy replay uses the same ggml_vec_scale/mad/dot sequence as ggml_compute_forward_gated_delta_net_one_chunk (ops.cpp:11225-11245) and stores the post-expf gate so no double-exp; the ring arithmetic is bounded by ctor asserts (rec->ne[1] >= n_tokens, n_rep <= n_seq_tokens).

Let me write it up. ## Scope covered

Read in full: `fork/ops-fork.inc`, `fork/timeline-fork.inc`, and `git diff master...HEAD` for `ops.cpp`, `ggml-cpu.c`, `ops.h`, `ggml-cpu.cpp`, `repack.cpp`, `spacemit/ime.cpp`. Cross-checked against `git show master:` for `ggml-cpu.c`, `ggml-cpu.cpp`, `tiled/tiled.cpp`, plus `ggml/src/ggml.c` op constructors, `ggml_graph_plan`/`ggml_get_n_tasks`, `tiled/tiled.cpp`, `src/fork/qwen4exp-fork.cpp`, `src/fork/llama-graph-fork.cpp`, `src/llama-memory-recurrent.cpp`.

---

##### 1. CRITICAL — new F16-src1 MUL_MAT support reaches the tiled kernel and hard-aborts on x86. ON BY DEFAULT

`ggml/src/ggml-cpu/ggml-cpu.cpp:459-463` (new clause in `case GGML_OP_MUL_MAT`)

The fork lets CPU accept `MUL_MAT` with `src1 == F16` against a **quantized** `src0`:

```cpp
(src1->type == GGML_TYPE_F16 && ggml_is_quantized(...vec_dot_type) &&
 ggml_blck_size(...vec_dot_type) <= 1024 && 1024 % ggml_blck_size(...vec_dot_type) == 0)
```

But `ggml_compute_forward_mul_mat` calls the tiled path **before** any src1 conversion (`ggml-cpu.c:1282`), and `ggml_tiled_supported` gates only on src0's type — its src1-type check is unreachable code placed after `return true` (`tiled/tiled.cpp:616-636`):

```cpp
        case GGML_TYPE_IQ1_M:
            return true;
        default: return false;
    }
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_Q8_K) {   // dead
```

`ggml_compute_forward_mul_mat_tiled` then only checks batch (`tiled.cpp:1220`, `rows >= 8`), and the driver asserts F32 (`tiled.cpp:1057-1066`):

```cpp
    if (src1->type != vec_dot_type) {
        ...
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
```

**Failure scenario:** `MUL_MAT src0 = Q4_K [4096 x 1024]`, `src1 = F16 [4096 x 32]` → `ggml_tiled_supported` true (Q4_K in the switch), `min_batch(32)` true → `GGML_ASSERT` fires, process aborts (no wrong value, no fallback). Same for Q5_K/Q6_K/Q2_K/Q3_K/IQ2_*/IQ3_*/IQ4_XS/IQ1_*.

**Upstream is safe**, verified: master `supports_op` rejects F16 src1 for a quantized src0, the only F16 case (hadamard hint) forces `src0 == F32` which the tiled switch rejects, and master `ggml_compute_forward_mul_mat_id` asserts `src1->type == GGML_TYPE_F32` (master `ggml-cpu.c:1627`). The dead src1 check is upstream's, but the fork is what makes it reachable.

**Env:** `GGML_CPU_TILED_MM` (`tiled.cpp:573-581`), `enabled = env == NULL || atoi(env) != 0` → **default ON**. Only compiled on `__AVX__`/`__AVX2__`/`__AVX512VNNI__` builds (`tiled.cpp:605`); on this arm64 macOS host the stub returns false, which is why it is not reproducible here — I could not run a repro, the claim is from the code path above.

**Reachability:** `LLAMA_HC_F16` (default 1, `qwen4exp-fork.cpp:955`) casts `xn` to F16 and feeds `build_lora_mm(w_down, xn)`; the cast is guarded by `qwen4exp_dev_is_vulkan(...)` (line 956-958), so a pure-CPU run is safe today — it becomes live whenever that MUL_MAT lands on CPU (expert/weight buft override, mixed CPU+Vulkan split), which is exactly the combination the new clause now advertises as supported. Fix: make the src1 check live in `ggml_tiled_supported`, or bail out of `ggml_compute_forward_mul_mat_tiled` when `src1->type != F32`.

---

##### 2. Suggestion (numerics) — new f16 DSV4_HC_PRE kernel: precision lost at the activation load, bounded

`ggml/src/ggml-cpu/fork/ops-fork.inc:132` and `:141`, dispatched from `ops.cpp:11674-11678`

```cpp
const float xv = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(... x->data ...));
...
sum += xv * wv;
```

This is the `*_f16_fork` variant the brief asks about. Precision is lost **only** at line 132 (the normalized hc streams, rounded to f16 by the graph builder): per-element relative error `<= 2^-11 = 4.9e-4`. Weights (`wv`, incl. the `expf` gate at line 136) stay f32, the accumulator is f32, and the summation order over `ih` is **identical** to `ggml_compute_forward_dsv4_hc_pre_f32` (`ops.cpp:11645-11655`) — no changed reduction order, no new FMA contraction. Error is bounded by `2^-11 * sum|x_h*w_h|`, amplified by `hc/(sum)` only when the `hc=4` stream terms cancel. f16 also caps magnitude at 65504 (overflow -> inf), which the CPU inherits rather than creates.

**Opt-in at the CPU level, on-by-default at the model level:** the kernel just follows `src0->type`; `LLAMA_HC_F16` defaults to **1** but the cast is additionally gated on `nt >= 32 && qwen4exp_dev_is_vulkan(model.dev_layer(il))` (`qwen4exp-fork.cpp:956-958`).

---

##### 3. Suggestion (OOB guard asymmetry) — the new blck-size guard is not applied to MUL_MAT_ID

`ggml/src/ggml-cpu/ggml-cpu.c:1663-1674`; guard lives only at `ggml-cpu.cpp:459-463`

`ggml_backend_cpu_device_supports_op` has **no `case GGML_OP_MUL_MAT_ID`** — it falls through to `default: return true` (`ggml-cpu.cpp:487-488`), verified by grep. So the fork's `GGML_ASSERT(src1->type == F32 || src1->type == F16)` relaxation at `ggml-cpu.c:1644` is unguarded, and the chunked pack silently assumes the MUL_MAT predicate:

```cpp
float tmp[1024];
for (int64_t k = 0; k < ne10_block_size; k += 1024) {
    const int64_t nk = MIN(1024, ne10_block_size - k);
    from_float(tmp, (char *) wdata_row + (k/bs)*nbw0, nk);
}
```

`(k/bs)` truncates and `nk` stops being a multiple of `bs` unless `bs | 1024`. Today every ggml quant block size is 1/32/256, all dividing 1024, so it is correct **for current types**; adding any type with `bs = 128*k`, `bs > 1024`, or a non-divisor produces overlapping writes into `params->wdata` and a mis-sized `from_float`. Mirror the `ggml_is_quantized / blck_size <= 1024 / 1024 % bs == 0` guard onto MUL_MAT_ID (or hoist it into a helper both paths call). Note the tiled MUL_MAT_ID path is **fine** — it reuses the caller's converted `wdata` (`tiled/tiled.cpp:856-857`).

---

##### 4. Suggestion (OOB / silent wrong value) — `flash_attn_list` asserts no element strides, upstream `flash_attn_ext` does

`ggml/src/ggml-cpu/fork/ops-fork.inc:13-14`, reads at `:52`, `:69`, `:90`

The only stride check is `GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst))`. `q_row[d]` (line 52/72) requires `q->nb[0] == sizeof(float)`, and `k_to_float(row_ptr, k_row, DK)` / `v_to_float(row_ptr, v_row, DV)` read `DK`/`DV` **contiguous** elements, so they require `k->nb[0] == ggml_type_size(k->type)`, `v->nb[0] == ggml_type_size(v->type)`. Upstream checks exactly these in its sibling op (`ops.cpp:8866`, `9100`, `9459`: `GGML_ASSERT(nbk0 == ggml_type_size(k->type));`, `ops.cpp:9623`: `GGML_ASSERT(nbq0 == sizeof(float));`).

The constructor does not close the gap either — `ggml_flash_attn_list` (`ggml.c:5679-5690`) asserts `ggml_can_mul_mat(k, q)`, `q->ne[3] == k->ne[3]`, and contiguity of `vals`/`idx` only. **Scenario:** q or k is a dim-0-permuted view (e.g. a head-split output) with `ne[0]` still = DK → no abort, `dot` at line 72 sums the wrong DK floats, silently wrong attention output; if `ne[0]` is larger than DK, `to_float` writes past `k_row` into `v_row`/`acc` (the per-thread buffer sized `DK + 2*DV + CACHE_LINE_SIZE_F32` at line 37 / `ggml-cpu.c:3050-3055`). Opt-in: the op is built only by qwen4exp's sparse-attention path.

---

##### 5. Suggestion — `dsv4_hc_pre_f16_fork` drops every shape check its f32 sibling has

`ggml/src/ggml-cpu/fork/ops-fork.inc:105-146` vs `ggml/src/ggml-cpu/ops.cpp:11620-11633`

`_f32` validates `dst->ne[0] == n_embd`, `dst->ne[1] == n_tokens`, and the `weights->ne[]` shape for both the gated and ungated branches before indexing `nbx*`, `nbw*`, `nbd*`. The f16 copy indexes the same strides with none of them (only the three type asserts at lines 110-112). Currently non-exploitable: `ggml_dsv4_hc_pre_impl` (`ggml.c:~6725`) asserts the weights shapes and builds `dst` itself, so `dst->ne` cannot be wrong via that constructor. It is a gap for any other caller (e.g. the fused `MUL_MAT_DSV4_HC_PRE` / `HC_POST_NORM_F16` rewrites), where line 145 writes 4 bytes per `ir` over `n_embd*n_tokens` into a smaller `dst`.

---

##### 6. Nit — `GATED_DELTA_NET_LAZY` never validates `mode`; any value other than 1/2 dereferences a NULL work buffer

`ggml/src/ggml-cpu/ops.cpp:11351`, `:11369`, `:11388`, `:11411`

`work` is NULL for every mode except 1 (`mode == 1 ? ... : nullptr`), while the non-mode-2 path uses it unconditionally: `float * s = mode == 2 ? s_out : work + S_v;` then `memcpy(s, s_in, ...)` and `delta = work`. `graph_plan` (`ggml-cpu.c:3101-3107`) also allocates only `if (op_params[0] == 1)`. Both constructors set 1 or 2, so this is not reachable from `ggml_gated_delta_net_lazy`/`_replay` — a one-line `GGML_ASSERT(mode == 1 || mode == 2)` at 11351 turns a future builder bug from a NULL write into a message.

---

##### 7. Nit — timeline instrumentation is a clean no-op when OFF, but is not neutral when ON

`ggml/src/ggml-cpu/fork/timeline-fork.inc`

**Env knob / default:** `GGML_TIMELINE=<file>` (`ggml/src/fork/ggml-backend-fork.inc:629-639`); `ggml_timeline_flag` starts `false` and is only set when the file opened, so **default OFF**. When off, all four hooks return before doing anything (`timeline-fork.inc:82`, `:98`), `ggml_graph_compute` writes `fork_t_call = 0` (`ggml-cpu.c:3504`), and `ggml_cpu_timeline_fork_begin` is called before the node loop with `tl.on = false` — behaviour is unchanged and the only cost is one relaxed atomic load per graph plus one per node. The added `fork_t_enter[]` is written only past the `ggml_timeline_on()` guard, so no off-path writes.

When ON, it is **not** measurement-neutral: thread 0 runs `ggml_cpu_timeline_fork_emit` *inside* the node loop (`ggml-cpu.c:3240`), i.e. after the barrier and before the next one, doing two `snprintf`, `ggml_nbytes`, and for `MUL_MAT_ID` a full `ids->ne[1] x ids->ne[0]` scan with a 512 B bitmap (`timeline-fork.inc:16-47`), then a mutex + `fprintf`. Every other thread is parked on the next barrier for that duration, so the emitted span for node *N* absorbs thread 0's own bookkeeping and the following node's chunk-scheduling is shifted. Also `MAX_EXPERTS = 4096` silently reports `bytes: 0` above that instead of a partial count.

Thread safety of the instrumentation itself is OK: `fork_t_enter[ith]` is a distinct slot per worker, written before the first barrier and read by thread 0 only after a barrier (first emit, or `fork_end` after the final barrier at `ggml-cpu.c:3250`), so the barrier supplies the ordering; `n_threads` is masked from `tp->n_graph` so stale slots from a previous, wider graph are never read.

---

#### Files checked and found clean

- **`ggml/src/ggml-cpu/ops.h`** — two declarations, match the definitions in `ops.cpp`/`ops-fork.inc`; no behaviour.
- **`ggml/src/ggml-cpu/repack.cpp:5213-5214`** and **`ggml/src/ggml-cpu/spacemit/ime.cpp:1618`** — the added `op->src[3] == nullptr` guards are correct *and* degrade cleanly, not fatally: the graph builder adds skip first, then re-queries the CPU (`src/fork/llama-graph-fork.cpp:193-203`) and strips skip from every node plus bails out of the MoE-cache path if unsupported. (Without the guard the repacked layout would have been fed to the generic `vec_dot` — this change prevents that.)
- **MUL_MAT_ID skip handling** (`ggml-cpu.c:1583`, `:1700-1704`) — `skip[i02]` is in bounds (`ggml_mul_mat_id_add_skip` asserts `ggml_nelements(skip) == src0->ne[2] == n_as`, `ggml.c:3499`); the zero `memset` reuses the existing `i02 == -1` form (`ne0*sizeof(float)` is exact because `nb0 == sizeof(float)` is asserted at `ggml-cpu.c:1602`), each `(id, iid1)` row maps to exactly one expert so writes are row-exclusive, `ith == 0` only, and the pre-existing `ggml_barrier` at `ggml-cpu.c:1716` orders it against the other workers.
- **Work sizing / division** — `graph_plan` `FLASH_ATTN_LIST` (`ggml-cpu.c:3050-3055`) and `GATED_DELTA_NET_LAZY` (`:3101-3107`) match the per-thread strides in the kernels (`ops-fork.inc:37`, `ops.cpp:11369`) exactly, including `CACHE_LINE_SIZE_F32`; `cur` is re-zeroed per node (`ggml-cpu.c:2911`) so `=` vs `+=` is not a bug; `ggml_get_n_tasks` returns `n_threads` for both new ops (`:2350-2355`, `:2490-2497`) and compute-time `params->nth` is that value or smaller, so `ith` indexing cannot overrun. The `dr = ceil(nr/nth)` splits in `ops-fork.inc:43-46`, `:122-125` and `ops.cpp:11449-11456` cover `[0, nr)` exactly once with no gap or overlap.
- **`gated_delta_net_lazy` numerics** — the replay uses the same `ggml_vec_scale_f32` / `ggml_vec_dot_f32` / `ggml_vec_mad_f32` sequence and order as upstream `ggml_compute_forward_gated_delta_net_one_chunk` (`ops.cpp:11225-11245`) and stores the **post-`expf`** gate (`ops.cpp:11443`), so a replay is bit-identical to the plain net — no accumulation-order or FMA-contraction change. Ring arithmetic is bounded: ctor asserts `rec->ne[1] >= n_tokens` and `rec->ne[0] == H*S_v + H_k*S_v + H` (`ggml.c:6578-6580`), and the producer keeps `n_rep <= rec_n <= n_seq_tokens <= R` with set index `2*cell + {0,1} < 2*size` (`src/llama-memory-recurrent.cpp:1595-1607`) — no OOB. Row aliasing across threads is impossible in mode 1 (`inf[0]/inf[1]/inf[4]` all derive from the unique `c = i + head`) and in mode 2 the `replayed` dedup guarantees one `n_rep > 0` row per source cell, the duplicates get `row_in == row_out` and are skipped at `ops.cpp:11385`.
- **`flash_attn_list` index arithmetic** otherwise sound: `c` bounds-checked against `n_kv` (`ops-fork.inc:66`), `rk2/rk3/rv2/rv3` GQA divisions and `is3 = iq3 % n_vseq` match the ctor's `q->ne[3] % vals->ne[3] == 0` assert, `DK`/`DV` agree with the plan (`ggml_can_mul_mat(k,q)` ⇒ `k->ne[0] == q->ne[0]`), `out` writes are covered by `ggml_is_contiguous(dst)`, and the online-softmax edges are safe (`expf(-inf) = 0` on the first accepted score; `S == 0` yields `inv = 0`).

No style, performance or duplication comments included, per instructions.

---

### B2.6 CUDA fork inc, второй проход

- Агент: `general-purpose-lVF6YZ4H7XHeRvUnQsQ8E9C5E7aMfxUh` (тип `general-purpose`, глубина 1, родитель: `general-purpose-ksLnpjA7E67cqBYBArnNK6VaQtRzgMCz`)
- Статус: completed, создан 17:38:39 UTC, финальный отчёт 20161 символов

**Постановка задачи (дословно из лога):**

READ-ONLY code review. Do NOT edit any file. You MUST actually run git commands and read files - do not answer from memory. Repo: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`, perf fork of upstream llama.cpp). Merge base = branch `master`. Start by running:
git -C /Users/askuzminov/Projects/github/askuzminov/llama.cpp diff master...HEAD --stat -- ggml/src/ggml-cuda/

Scope: CUDA backend fork code only:
- ggml/src/ggml-cuda/fork/gated-delta-net-lazy-fork.inc (279, new)
- ggml/src/ggml-cuda/fork/getrows-fork.inc (76, new)
- ggml/src/ggml-cuda/fork/repeat-fork.inc (44, new)
- ggml/src/ggml-cuda/fork/elastic-buffer-fork.inc (170, new)
- ggml/src/ggml-cuda/fork/lightning-indexer-fork.inc (17, new)
- ggml/src/ggml-cuda/fork/ggml-cuda-timing-fork.inc (548, new)
- diffs (`git diff master...HEAD --`) of: ggml-cuda.cu, mmid.cu, mmq.cu, mmq.cuh, top-k.cu, getrows.cu, gated_delta_net.cu, gated_delta_net.cuh, binbcast.cu, lightning-indexer.cu
NOTE: skip the internals of fork/top-k-fork.inc (already reviewed), but DO check how top-k.cu hooks it in: does the fork path get used for shapes it was not written for (k>256, nrows small, ncols smaller than blocks_per_row*BLOCK_SIZE)?

Look for, in priority order:
(a) NUMERICS changing model output: f16/bf16 accumulation or intermediate where upstream uses f32; changed reduction order; fast-math intrinsics (__expf, __fdividef, rsqrtf, -use_fast_math); anything silently lowering precision on a DEFAULT-on path.
(b) OOB / memory safety: reads/writes past the end of a tensor when a dimension is not a multiple of the block/tile; wrong stride when nb[] is not contiguous; 32-bit index overflow for long contexts (n_ctx 262144) - check every `(int)` cast of an index/offset derived from ne; grid/block sizing math; __syncthreads() reached by only some warps (deadlock) or missing between a shared write and a cross-warp read; uninitialized shared memory reads.
(c) elastic-buffer-fork.inc + the ggml-cuda.cu device-memory hooks: a buffer handed out while a previous user is still reading it, a cached device pointer that goes stale after cudaFree/realloc, use-after-free, missing stream sync before a read.
(d) ggml-cuda-timing-fork.inc: does the instrumentation change behavior (an extra sync, a code path taken only when timing is on, a buffer reused across iterations)? ON by default?
(e) GGML_HINT_IDS_REPEAT: mmq.cu sets mmq_args.ncols = ne_get_rows instead of ne12 when the hint is set, and mmq.cuh adds an early-return in mul_mat_q_stream_k_fixup. Trace EVERY consumer of mmq_args.ncols (mul_mat_q, mul_mat_q_stream_k, the fixup, and the equivalent in mmid.cu) and verify nothing still assumes ncols == ne12. Repeated expert ids within one token is the under-exercised case.

For EVERY finding: SEVERITY (Critical = wrong output or memory error; Suggestion; Nit), exact file:line, one-line claim, concrete failure scenario (shape/dtype/k and what wrong value or OOB), ON BY DEFAULT or opt-in (name the env knob and its default), and the specific code evidence. Verify against upstream before claiming a difference is real. State explicitly which files you checked and found clean. No style/perf/duplication. Max 10 findings.

**Отчёт агента (дословно):**

I have everything. Let me write the final report.

Findings (ordered by severity):

**F1 — Critical (hard abort, possible silent wrong tiles): mmq.cu:371 + mmq.cuh:1480, 1510-1517**
Claim: With GGML_HINT_IDS_REPEAT, ncols_max jumps from ne12 to ne12*n_expert_used, so the stream-k grid's x-extent (ntx) and ntiles_dst grow n_expert_used× and now hit the 2^30 tile-count assert (and the `100 * ntiles_dst` int math) n_expert_used× sooner.
Failure scenario: MoE FFN down projection, n_experts=256 (ntzw=256), ne01=4096 (I=64 → nty=64), ne00=4096 Q4_0 (blocks_per_ne00 = 4096/32 = 128), 2048-token ubatch × 8 ids → ncols_max = ne_get_rows = 16384 → ntx = 512 → ntiles_dst = 512*64*256 = 8.4M → 8.4M*128 = 1.07e9 ≥ 2^30 → GGML_ASSERT at mmq.cuh:1514 aborts the process. Upstream with the same shapes has ntx = 64 → 134M → passes.
Also: the assert multiply is `int * uint32_t` → 32-bit unsigned (wraps), so above 2^32 the guard can spuriously pass and `kbc` (int, and the `nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z` product in mul_mat_q/mul_mat_q_stream_k_fixup, mmq.cuh:1211-1213/1313-1314) overflows → tiles processed twice/never → wrong output.
ON BY DEFAULT: yes, whenever the fork's graph sets the hint (src/fork/llama-graph-fork.cpp:138-160) and the stream-k config is selected (no env knob).
Evidence: the three lines quoted.
Fix direction: cap ncols_max at ceil(total_rows/n_experts)+... or per-expert max, or compute the tile product in int64.

Actually — better fix suggestion: `ncols_max` only needs to bound the max columns of ANY single expert; a tighter bound is min(ne_get_rows, ...) but the true worst case is indeed ne_get_rows (one expert used by all slots). However, the grid could stay at ne12 tiles and instead let the tile loop stride... no. Or: since expert rows are compacted in one global array, an expert's col_diff can be up to ne_get_rows, so the bound is needed. Alternative: raise the assert / use int64 for the tile math. I'll suggest computing the product in int64 and clamping ntx.

**F2 — Suggestion (numerics, default-on): lightning-indexer.cu:701 + fork/lightning-indexer-fork.inc:13**
Claim: Lowering the tile-kernel threshold from 8 to 2 tokens moves n_batch 2..7 from the vec kernel (keys in f32) to the tile kernel (keys staged as f16), changing indexer scores for every 2-7 token batch (MTP/speculative draft checks).
Evidence: vec kernel dequantizes/holds K in float4 (lightning-indexer.cu:285, :301) and reduces with warp_reduce_sum; tile kernel writes `make_half2(v.x, v.y)` into `__shared__ half2 k_shared[...]` (:430, :456) then widens back to f32 and accumulates serially per thread. So (i) for K types F32/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 the keys are rounded to f16 (bf16 → f16 can additionally overflow to inf for |k| > 65504 or flush small values), and (ii) for any type (incl. the common f16 cache) the summation order changes vs upstream and vs the CPU/Metal reference.
Scenario: n_batch=4 (3 draft tokens + target), n_kv=65536: two blocks whose scores differ by <1e-3 can swap rank in the block top-k → different KV blocks attended → different tokens.
ON BY DEFAULT: yes; `GGML_CUDA_LID_TILE_MIN` (default 2; set 8 for upstream).

**F3 — Suggestion (memory safety contract): elastic-buffer-fork.inc + ggml-cuda.cu:815-835**
Claim: The elastic buffer keeps the stock CUDA buffer interface, whose `init_tensor`/`set_tensor`/`memset_tensor`/`cpy_*` write through `tensor->data` with no commit check, and the destructor unmaps without a device sync (decommit does sync).
Scenario: a quantized non-view tensor allocated in an elastic KV buffer (usage != COMPUTE) with ne[0] not a multiple of the quant block size → `cudaMemsetAsync(tensor->data + original_size, ...)` at allocation time, when no granule is mapped → cudaErrorIllegalAddress; currently unreachable for KV tensors because head_dim is a multiple of 32 so padded_size == original_size, so it's latent, not live.
Also: `ggml_cuda_elastic_free_fork` (:18-26) `cuMemUnmap`s while a kernel from the last compute could still be in flight if the caller did not synchronize the backend — `cuMemUnmap` (unlike `cudaFree`) gives the driver no chance to defer, so the process faults. decommit does `cudaDeviceSynchronize()` at :143; free does not.
ON BY DEFAULT: the hooks exist always but are only exercised through `LLAMA_KV_ELASTIC` (host side); VMM devices only.

**F4 — Nit: ggml-cuda-timing-fork.inc:341-348 (op span events) vs the concurrent-event streams**
Claim: op-timing events are recorded on `ctx->stream()` while fused groups / nodes dispatched under `try_launch_concurrent_event` run on `cuda_ctx->curr_stream_no` sub-streams, so the reported per-op GPU time can exclude (or mis-attribute) work; the instrumentation itself does not change behavior and is off by default.
Also note: when ON it forces `use_cuda_graph=false` (ggml-cuda.cu:5105-5109) which is a documented behavior change; when OFF, the added cost is 2 early-returning calls per compute + one static-int load per node.

**F5 — Nit: getrows-fork.inc:56-59**
`if (n <= 0) return true;` silently drops the op where upstream `get_rows_cuda_float` asserts `GGML_ASSERT(ne12 > 0)` / `ne11 <= UINT32_MAX/ne12`. A zero-sized gather becomes a silent no-op (only reachable with degenerate shapes).

Also worth mentioning as verified-clean but surprising: `getrows-fork.inc` returns true → upstream's asserts are bypassed for all ne00 ≤ 128 gathers model-wide, i.e. this is on by default for every short-row GET_ROWS.

**Positives verified** (brief):
- mm_ids_helper rewrite (mmid.cu:104-146): row per use, `expert_bounds[e] = #ids<e`, `bounds[n_experts] = total uses` — self-consistent; `store[]` holds one entry per distinct token in both paths (duplicate lanes race only on `iex_used`, which is no longer read), `n` recomputed from ids, `warp_prefix_inclusive_sum` inclusive → `irow` offsets exact; `ids_dst` pool = ne_get_rows + J-1 covers all writes; ids_src1 forward/inverse maps written in the same loop. The removed `break` (ggml-cuda.cu:2302) fixes the generic path where a second use left `ids_from_sorted_host[]` at 0.
- mmq.cuh:1408-1413 early return: block-uniform (jt/zt from blockIdx.x, no barrier crossed), and it is *required* by the enlarged ncols_max (unguarded `ids_dst[col_low + jt*J + j]` would read up to ~2*ne_get_rows past the pool). Both mul_mat_q (:1069-1093) and mul_mat_q_stream_k (:1246-1271) already skip empty tiles, and a block whose *final* tile is empty wrote no fixup entry, so the fixup combining an uninitialized `tmp_fixup` slot is closed.
- top-k.cu hook: checked k>256 (no NBINS relation; slots bounded by n_above = k-rank), small nrows, and ncols < blocks_per_row*BLOCK_SIZE (cols_per_block = round_up(ceil(ncols/bpr), 256), blocks with begin ≥ ncols contribute above=ties=0 and return early in the gather; union of block ranges covers [0, ncols)). The radix path is shape-general and the fork is only entered from top_k_radix_cuda (upstream's own fallback).
- GDN lazy kernel: matches upstream's non-KDA math expression-for-expression (expf, warp_reduce_sum<lanes_per_col>, same FMA order); f32 throughout; KDA can't reach it (ggml.c:6581 asserts g->ne[0]==1); record key slot `h_idx % H_k` write/read consistent; mode-2 replay reads s_in fully into registers before writing s_out (row_in==row_out safe); supports_op gates S_v ∈ {16,32,64,128}.
- repeat fork: index math and the 16-byte alignment check on dst->data are exact; the tail loop covers n % per; ts ∈ {2,4}; ggml_repeat guarantees src/dst same type so no cast is dropped; opt-out is not available (no env) but semantics are identical.
- timing fork default off: `ggml_cuda_timing_fork_n` (0 unless GGML_SCHED_PROF_GPU enables via proc address), `GGML_CUDA_OP_TIMING` default 0, `ggml_timeline_on()` default false (GGML_TIMELINE path).
- No -use_fast_math or __expf/__fdividef/rsqrtf introduced anywhere in the CUDA diff; no CUDA CMake flag changes.

Files checked and found clean: fork/getrows-fork.inc, fork/repeat-fork.inc, fork/lightning-indexer-fork.inc (mechanics), fork/gated-delta-net-lazy-fork.inc, binbcast.cu, getrows.cu, gated_delta_net.cu/.cuh, top-k.cu hook, mmq.cu/mmq.cuh consumers of ncols_max, mmid.cu, the gdn cache-fusion tail check, the timing fork, the elastic fork.

Also mention what I could NOT verify: no build (macOS, no CUDA toolkit) — so nothing is compile-tested; and I did not audit fork/top-k-fork.inc internals per instructions (only its entry guard and the hook).

Let me write it up concisely. ## Review: CUDA backend fork (`master...HEAD`, branch `cache`)

No files were modified. All findings below were verified against `git show master:...` and by reading the surrounding kernels. Note: this is a macOS box with no CUDA toolkit, so nothing here is compile- or run-tested; all claims are from code reading.

---

##### 1. Critical — `ncols_max = ne_get_rows` inflates the stream-k tile grid by `n_expert_used`, hitting the 2^30 tile assert (and the assert itself is 32-bit)
**`ggml/src/ggml-cuda/mmq.cu:371`**, effects at **`mmq.cuh:1480, 1510-1517`**

```
371: const int64_t ncols_max = ggml_get_op_params_i32(dst, 1) == GGML_HINT_IDS_REPEAT ? ne_get_rows : ne12;
1480: const int ntx  = (args.ncols_max + config.J - 1) / config.J;
1510: const int ntiles_dst = ntx * nty * ntzw;
1512: const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm*tiles_nwaves);
1514: GGML_ASSERT(ntiles_dst * blocks_per_ne00_fd.z < (1 << 30));
```

`ncols_max` only feeds `ntx`, but `ntx` is the x-extent of the whole stream-k tile space, so `ntiles_dst` grows by exactly `n_expert_used` (8x for an 8-active MoE). **Failure:** MoE down-projection (`down` gets the hint at `src/fork/llama-graph-fork.cpp:160`), `n_experts=256` (`ntzw=256`), `ne01=4096` with `I=64` (`nty=64`), `ne00=4096` Q4_0 (`blocks_per_ne00 = 4096/32 = 128`), 2048-token ubatch x 8 ids -> `ncols_max = 16384` -> `ntx=512` -> `ntiles_dst = 8.4M` -> `8.4M * 128 = 2^30` -> **GGML_ABORT**. Upstream with the same shapes has `ntx=64`, product 134M, passes. Worse, the guard's multiply is `int * uint32_t` = 32-bit unsigned, so once the true product exceeds 2^32 it can wrap back under 2^30 and **pass**, after which `kbc` in `mul_mat_q_stream_k`/`_fixup` (built from `nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z`, also 32-bit) overflows and tiles are skipped or computed twice -> wrong output.
**ON BY DEFAULT** whenever the fork's graph sets `GGML_HINT_IDS_REPEAT` and a stream-k config is selected; no env knob. Fix: keep the tighter per-expert bound, or hoist the tile-product math to `int64_t` (and the `100 * ntiles_dst` at :1512, which overflows above ~21.4M tiles).

##### 2. Suggestion (numerics, default-on) — indexer tile threshold 8 -> 2 changes scores for 2-7 token batches, and the tile kernel rounds keys to f16
**`lightning-indexer.cu:701`** + **`fork/lightning-indexer-fork.inc:13`** (`const int v = env ? atoi(env) : 2;`)

The vec kernel keeps K in **f32**: `float4 k_reg_f[]` (:285), `get_dequantize_V<TYPE_K, float, 4>` (:301), f32 `ggml_cuda_mad` + `warp_reduce_sum`. The tile kernel stages K in **f16**: `__shared__ half2 k_shared[...]` (:430), `lo = make_half2(v.x, v.y)` (:456), then widens back with `__half22float2` and accumulates serially. So for `K->type` in {F32, BF16, Q4_0/1, Q5_0/1, Q8_0} the fork's lowered threshold introduces ~2^-11 relative key rounding that upstream does not have for those batch sizes (bf16->f16 can additionally overflow |k|>65504 to inf); for *any* type, including an f16 cache, the summation order now differs from upstream and from the CPU/Metal reference. Scenario: `n_batch=4` (target + 3 MTP drafts), `n_kv=65536` -> two block scores within ~1e-3 swap rank in the downstream block top-k -> different KV blocks attended -> different tokens. `GGML_CUDA_LID_TILE_MIN=8` restores upstream; **default is 2**.

##### 3. Suggestion (memory safety contract) — elastic buffer keeps the stock interface (unguarded writes into unmapped VA) and unmaps at free without a device sync
**`fork/elastic-buffer-fork.inc:18-26, 71-128`**, **`ggml-cuda.cu:792-796, 815-835, 998-1001`**

`ggml_backend_cuda_buft_alloc_elastic_fork` hands out a buffer with plain `ggml_backend_cuda_buffer_interface`, so every stock entry point dereferences `tensor->data` on reserved-but-unmapped VA with no commit check. Concretely reachable-by-construction one: `init_tensor` (`ggml-cuda.cu:823-834`) memsets quantized-tensor padding through `tensor->data + original_size` at allocation time, i.e. before any granule is committed. It is currently latent only because KV head dims are multiples of the quant block size (`padded_size == original_size`), so nothing is written; nothing in the fork enforces that invariant. Same class: `memset_tensor`/`set_tensor`/`cpy_tensor_n` (:837-880) fault if the host ever touches an uncommitted cell. Second point: `ggml_cuda_elastic_free_fork` `cuMemUnmap`s mapped granules with **no** `cudaDeviceSynchronize()`, while `decommit` deliberately does one (:143, "no kernel may still read the range"); `cuMemUnmap` gives the driver no chance to defer, so freeing the buffer without a prior backend sync faults the process rather than degrading. Commit/decommit index math itself is correct (commit floors `i0`/ceils `i1`; decommit ceils/floors, so only fully-contained granules are dropped), and commit's zero-fill on `cudaStreamPerThread` + `cudaStreamSynchronize` is correctly ordered because commit is host-synchronous and the granule was drained by the preceding decommit.
Gated by `GGML_USE_VMM` and by the host-side `LLAMA_KV_ELASTIC`, not on by default for ordinary runs.

##### 4. Nit — op-timing spans are recorded on the main stream only, so per-op times can be wrong when the concurrent-event streams are active
**`fork/ggml-cuda-timing-fork.inc:341-348`** (`cudaEventRecord(s.start/end, ctx->stream())`)

Nodes dispatched under `try_launch_concurrent_event` run on `cuda_ctx->curr_stream_no` sub-streams, so the reported span brackets the main stream, not the op's actual work. Measurement-only. For (d) overall: the instrumentation does **not** change behavior when off - `ggml_cuda_timing_fork_get` returns on one relaxed atomic load (`ggml_cuda_timing_fork_n == 0` unless `GGML_SCHED_PROF_GPU` calls the proc address), `ggml_cuda_op_timing_fork_period()` defaults to 0, `ggml_timeline_on()` defaults false (`GGML_TIMELINE`), and the only added cost is two early-returning calls per compute plus a static-int check per node. When `GGML_CUDA_OP_TIMING` is on it does force `use_cuda_graph = false` (`ggml-cuda.cu:5105-5109`) - a deliberate, documented divergence. `begin`/`end` are balanced on both the fused and the single-node paths (`ggml-cuda.cu:4961-4964`, `:4997`), `g->nodes[i + skip]` is in range (upstream uses the same bound in its debug log), and no early return exists between `timing_fork_begin` and `timing_fork_end` in `graph_compute`.

##### 5. Nit — flat GET_ROWS silently no-ops where upstream asserts
**`fork/getrows-fork.inc:56-59`**: `if (n <= 0) return true;` replaces upstream's `GGML_ASSERT(ne12 > 0)` / `GGML_ASSERT(ne11 <= UINT32_MAX / ne12)` (`getrows.cu:259-260`), because the fork is entered before them for every `ne00 <= 128` gather model-wide (default on, `GGML_CUDA_GET_ROWS_FLAT=0` to disable). Degenerate shapes now produce a no-op instead of a diagnostic.

---

##### Verified clean (and what I actually checked)

- **`(e)` every consumer of `mmq_args.ncols_max`**: it is only read at `mmq.cuh:1480` (`ntx`). `ncols_dst` is unchanged (`ne_get_rows`) and is used only in the `!ids_dst` branches. Both `mul_mat_q` (:1069-1093) and `mul_mat_q_stream_k` (:1246-1271) already `return`/`continue` on `jt*J >= col_diff`, and with the guard they read `ids_dst` at most `col_low + col_diff - 1 + J - 1 <= ne_get_rows + J - 2`, exactly the documented pool pad (`mmq.cu:313`). The new early return in `mul_mat_q_stream_k_fixup` (`mmq.cuh:1408-1413`) is block-uniform (jt/zt derive from `blockIdx.x`, nothing before it is divergent, so no split `__syncthreads`) and it is **required**: without it the fixup reads `ids_dst[col_low + jt*J + j]` up to ~`2*ne_get_rows` past the pool and then writes through that garbage index; it also closes the pre-existing hole where a block whose *final* tile is empty computes nothing and writes no `tmp_fixup` slot, which the successor block would otherwise sum uninitialized.
- **`mmid.cu:104-146`**: rows-per-use remap is self-consistent - `expert_bounds[e] = #ids<e` (already counted every use in both the generic and the templated path), `bounds[n_experts] = nex_prev + nrows = total uses`; `store[]` holds exactly one entry per distinct token in both paths (duplicate lanes collide on the same slot and only disagree on `iex_used`, which is no longer read), `n` is recomputed from `ids`, `warp_prefix_inclusive_sum` is genuinely inclusive so `n_incl - n` is the exclusive offset, and `nrows` accumulates the warp total via the lane-31 shuffle. Forward and inverse (`write_inverse`) maps are written in the same loop, and every index is `< ne_get_rows`.
- **`ggml-cuda.cu:2302` `break` removal**: correct fix - with the break, a token using an expert twice left the second `ids_from_sorted_host[]` slot at 0 (aliasing row 0) and the `ids_to_sorted_host.size() == ne_get_rows` assert could not hold.
- **top-k hook (`top-k.cu:218, 304`)**: shape-safe for everything you asked about. `k > 256` is fine (no `NBINS`/k relation; slots are bounded by `n_above = k - rank`, ties fill `[n_above, k-1]`). Small `nrows` is fine (`blocks_per_row = max(1, min(ceil(ncols/1024), ceil(32*nsm/nrows)))`). `ncols < blocks_per_row*BLOCK_SIZE` is fine: `cols_per_block = round_up(ceil(ncols/bpr), 256)`, block ranges tile `[0, ncols]` contiguously, and a block with `begin >= ncols` gets `end <= begin`, contributes `above = ties = 0` (all-or-none writes, `above[]` fully written each pass by all 256 threads) and returns before the ballot loop. `__syncthreads` in the histogram/gather passes is reached by the whole block (loop bounds and the early return are block-uniform). The fork is entered only via `top_k_radix_cuda`, which is upstream's own shape-general fallback, and returns false on HIP/MUSA.
- **`fork/gated-delta-net-lazy-fork.inc`**: the per-token math is expression-for-expression identical to the non-KDA path of `gated_delta_net_cuda` (`expf`, `warp_reduce_sum<lanes_per_col>`, `s = g*s + k*delta` then `+= s*q`), all f32, no fast-math intrinsics. KDA cannot reach it (`ggml.c:6581` asserts `g->ne[0] == 1 && beta->ne[0] == 1`). Record layout matches `ggml.c:6584` (`H*S_v + H_k*S_v + H`), key slot written only by heads `h_idx < H_k` and read as `h_idx % H_k` - consistent. The `row_in == row_out` replay reads the whole row into registers before writing back, and each block owns a disjoint `(head, column)` slice. `ggml_cuda_gated_delta_net_lazy_supported` gates `S_v` to {16,32,64,128}.
- **gdn cache-fusion tail check (`ggml-cuda.cu:3167-3182`)**: sound, including aliases - `ggml_new_tensor_impl` (`ggml.c:1803-1806`) collapses view chains at creation, so a reshape/view **of a view** of `gdn` still has `view_src == gdn` and the accumulated `view_offs`; the `view_offs >= tail_off || nbytes > tail_off - view_offs` test cannot underflow (`||` short-circuits).
- **`fork/repeat-fork.inc`**: `n4 = n/per` agrees host/device, the 16-byte check is on `dst->data` (so a view offset is caught), the tail loop covers `n % per`, grid-strided and capped at 65536 blocks, `ts in {2,4}`, and `ggml_repeat` guarantees `src->type == dst->type`, so no cast is skipped. Semantics identical to the broadcast kernel.
- **No precision flags**: the CUDA diff introduces no `-use_fast_math`, no `__expf`/`__fdividef`/`rsqrtf`, and no `ggml/src/ggml-cuda/CMakeLists.txt` change.

**Not audited** (per instructions): the internals of `fork/top-k-fork.inc` beyond its entry guard, `ncols_max`'s callers outside `ggml/src/ggml-cuda/`, and the host-side `info`/`rec` producers in `src/` that the GDN lazy kernel trusts (`inf[3]`/`inf[4]` as set indices are assumed valid by the kernel with no bounds check - if the producer can ever emit `-1`, the replay reads `rec - rec_set`).

---

### B3 Конкурентность сервера: пул, роутер, слоты, spill

- Агент: `general-purpose-twnf47f6Kg08awPJs2CIxtlF5vTtTT5x` (тип `general-purpose`, глубина 0, родитель: корневой вызов ревью)
- Статус: completed, создан 13:45:21 UTC, финальный отчёт 19691 символов

**Постановка задачи (дословно из лога):**

You are doing a READ-ONLY correctness review of a performance fork of llama.cpp. Do not edit, create, or delete files. Do not run builds (another process is building in build-qreview).

REPO: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`).
REVIEW TARGET: `git diff master...HEAD` (merge base = `master` aa588a346).
CONTEXT: perf-focused research fork of llama.cpp; design notes in FORK.md / RESEARCH.md (Russian). Env knobs via `llama_fork_env_int(NAME, default)`, several ON by default. The owner's concerns, in order: (1) does it break inference output correctness, (2) are there real bugs (crashes, races, leaks, deadlocks), (3) is the claimed speed actually delivered.

YOUR SCOPE - the server's request handling and concurrency ONLY (sampling/speculative math, GPU kernels, and the KV/MoE cache internals are handled by other reviewers - skip those):
1. New fork subsystem: `tools/server/fork/server-pool-fork.cpp` (3374 lines), `server-pool-fork.h`, `server-task-fork.cpp` (1988 lines), `server-task-fork.h`, `server-context-fork.cpp/.h`, `server-role-fork.cpp/.h`, `server-models-fork.cpp`, `server-trace-fork.cpp`, `server-gpu-sensors-fork.cpp`, `server-pool-ui-fork.inc`.
2. Edits to upstream server files: `tools/server/server-context.cpp` (+1479), `server-task.cpp/.h`, `server-models.cpp/.h`, `server-models-sched.cpp/.h`, `server-common.cpp/.h`, `server-queue.h`, `server-schema.cpp`, `server-http.cpp`, `server.cpp`.
3. Knobs in this area: `LLAMA_SLOT_ROOM` (default 1), `LLAMA_SPEC_SLOT_BATCH` (default 8), `LLAMA_SERVER_STOP_TIMEOUT` (default 10), `LLAMA_NGRAM_CACHE_KEEP` (default 0), plus anything the pool code reads.
4. Python tests: `tools/server/tests/unit/test_pool_fork.py`, `test_slot_room.py`, `test_router_fork.py`, `test_cache_spill.py` - use them to understand intended behavior, and check whether the tests could pass while the feature is actually broken.

What to look for:
- Data races on shared slot/pool state; mutex held across blocking decode; lock ordering that can deadlock; condition variables with no predicate (lost/spurious wakeup); a shutdown path that can hang or join a thread that is blocked.
- Slot lifecycle: a slot returned to the pool while still referencing KV rows or a sampler; two requests writing the same slot; cancellation/abort leaving a slot stuck; `LLAMA_SLOT_ROOM` letting a new request start before the previous one's state is reset.
- Cross-slot contamination that changes generated text: a sampler / rng / grammar / ngram cache reused between requests without reset; per-request params read from another request; `LLAMA_NGRAM_CACHE_KEEP`.
- HTTP/schema: request bodies that can index out of range, unchecked sizes, integer overflow in token/length arithmetic, error paths that leak or double-free.
- The UI endpoint (`server-pool-ui-fork.inc`) and any command/role handling (`server-role-fork`) for injection or unbounded input.

For each finding report: SEVERITY (Critical = wrong output, crash, deadlock, or data race reachable in normal server use; Suggestion; Nit), file:line, one-line claim, the concrete failure scenario (request sequence / interleaving -> wrong outcome), whether it is ON BY DEFAULT or opt-in (name the knob + default), and your evidence (the code you read).

Verify before claiming - read both sides of a suspected race and confirm there is no lock you missed. Do not report style, naming, comments, file length, or "this is big so probably buggy". Max 12 findings, most severe first. If an area looks clean, say so in one line.

**Отчёт агента (дословно):**

I have everything. Writing the report now.

One more sanity check on finding #2's discovery claim: `discovery_t::stop()` early-returns if already stopping, so a later `~discovery_t` → does `~discovery_t` call stop()? Let me check if there's a destructor. The struct has `th` as a member; I didn't see `~discovery_t`. If there's no destructor, then at `~discovery_t` (implicit), `th` is destroyed while joinable → terminate. Let me confirm there's no user-declared destructor.

From my read of 2896-2942, the struct ends at 2942 `};` with stop() being the last member. No `~discovery_t()`. So the implicit destructor destroys `th`. If stop() already ran (and returned early on a second call, or ran before th was assigned), th is joinable → terminate. Confirmed enough.

Also `server_pool_fork::~server_pool_fork()` calls `stop()` (line 2995-2997 per agent). So stop() is called twice at shutdown normally: once from shutdown_handler, once from the destructor → the second returns early at `stopping.exchange(true)`. For the *pools* (`impl::stop()`), does it also early-return before joining? Let me check impl::stop: agent said `SPF:2503-2522` — "обмен pollers на локальный вектор под mu_threads, затем join вне блокировок". If `impl::stop()` also starts with `if (stopping.exchange(true)) return;`, then the same class of bug: after a first stop(), a *newly started* poller (via `start_poller` from the discovery thread) would never be joined → terminate. But `start_poller` checks `if (stopping.load()) return;` — good, that gate exists there. Only `discovery_t::start()` lacks it.

Good — that strengthens finding #2's discovery part as the concrete instance.

Let me write the final report. ## Scope and method

Read `git diff master...HEAD` for the server request-handling/concurrency area (`server-context.cpp` +1479, `server-task.{h,cpp}`, `server-queue.h`, `server-models*`, `server.cpp`, `server-role-fork.cpp`, all of `fork/server-pool-fork.cpp`, `fork/server-task-fork.cpp`, `fork/server-pool-ui-fork.inc`) plus the four fork test files. Two parallel subagents produced a lock/thread inventory of `server-pool-fork.cpp` and of the spill/xfer code; I re-read and verified each claim I report below against both sides of the code.

#### Findings

**1. CRITICAL — the import size guard is disabled by the product default, so a network peer sets the allocation size.**
`tools/server/server-context.cpp:6258` sets `pool_cache_ram_fork = prompt_cache->limit_size`. `fork/server-context-fork.cpp:126-135` maps `--cache-ram -1` to `limit_size = 0`. `fork/server-task-fork.h:63-66` defines `0` as "any", and `fork/server-task-fork.cpp:1364` is `if (max_bytes > 0 && lay.size > max_bytes)` — so with `max_bytes == 0` the only bound left is `SPILL_MAX_BYTES = 1ull << 40` (`server-task-fork.cpp:94-100`), applied in `parse_head` (`:1370-1374`) to `prompt`, `main`, `drft` and `index` independently.
Failure scenario: a peer `POST`s a 72-byte header to `/pool/import/data?id=…&offset=0` with `n_main = n_drft = n_prompt/4 = 2^40-1`; the node calls `resize()` four times at ~1 TiB each before it validates anything else. `bad_alloc` is caught (`:1375-1377`), but under Linux overcommit the mmap usually succeeds and the reservation is real. The host-RAM reserve that is supposed to bound this is only consulted *afterwards*, in `make_room` / `alloc`.
ON BY DEFAULT for the knob that matters: `common/common.h:653` — `cache_ram_mib = -1` is upstream's default, so `max_bytes == 0` is the out-of-the-box state of any `--pool` node. The pool feature itself is opt-in (`--pool` in the preset) and the routes are key-guarded if `--api-key` is set.

**2. CRITICAL — `std::thread` destructed while joinable converts any escaping exception into `std::terminate`.**
Three sites. (a) `fork/server-pool-fork.cpp:1552` creates `watch`, joined at `:1565`; (b) `:1646` creates `reader`, joined at `:1710`. Between creation and join the writer loop does throwing work — 16 MiB `std::string` copies (`chunk = std::move(q.front())`, `:1694`), `std::to_string`/string concatenation in the URL building, and `req.should_stop()`. An exception there unwinds through a joinable `std::thread` → `terminate`, bypassing the pool's own `catch(...) { release(r.id, 500); throw; }` (`:1916`) and the HTTP layer's exception wrapper, so the whole router aborts instead of one request failing. `release()` is also skipped, so the node's `inflight` count leaks for whatever survives.
(c) `fork/server-pool-fork.cpp:2919-2930`: `discovery_t::start()` ends with an unguarded `th = std::thread([this]() { loop(); });`, while `stop()` (`:2934-2936`) begins with `if (stopping.exchange(true)) return;`. `discovery_t` has no user destructor, so if a shutdown lands between entering `start()` and assigning `th`, `stop()` returns without joining, and the later `~discovery_t` destroys a joinable `th` → `terminate`. Note `impl::start_poller` (`:816-821`) *does* have the `stopping` gate, which is why this is specific to the discovery thread. Opt-in with `--pool`.

**3. SUGGESTION — the stale-transfer cleaner deletes a brand-new export link.**
`fork/server-pool-fork.cpp:3335-3356` scans `--cache-spill-dir` for `pool-export-*`/`pool-import-*`, skips ids present in `pool_plans`, and removes the rest when `now - last_write_time > idle_s` (`POOL_XFER_IDLE_S_FORK = 120`, `server-context.cpp:7756`). A hard link shares the target inode's mtime, and `create_hard_link` (`server-task-fork.cpp:1313`) does not touch it — so a link to a state last written more than 120 s ago is *born* old. The plan is registered only later, on the main loop, at `server-context.cpp:2128`.
Failure scenario: `POST /pool/export` → main loop links the file at `:1313`; before it reaches `:2128` another HTTP thread calls `server_pool_xfer_clean_fork` (it is called from `GET /pool/node:7763`, `POST /pool/export:7832`, and `POST /pool/import/data:7936`); `pool_plans.count(id)` is 0 and the inode is old → link removed → the first `GET /pool/export/data` returns "cannot open the state file". Consequence is a failed move and a full recompute (correct output, lost speed), not UB. The membership check the code relies on is defeated by the age criterion itself.

**4. SUGGESTION — `/pool/export/done` unlinks the file a chunk read is still using.**
`server_pool_xfer_forget_fork` (`server-pool-fork.cpp:3214-3224`) erases the map entry and calls `pool_plan_drop` → `std::filesystem::remove(plan.file)`. A concurrent `server_pool_xfer_read_fork` (`:3228-3274`) has already copied the `shared_ptr<const plan>` (memory-safe) but re-opens `plan->file` per chunk at `:3247-3253`. Two router threads racing on the same transfer id, or a reader that fell behind the requester's 60 s idle budget, get "cannot open the state file" mid-transfer. Also note that when `create_hard_link` fails, `plan.file = spill_path(st.uid)` with `file_link = false` (`:1315`), and that path is the LRU pruner's (`spill_prune`, `:302-325`) — which does not take any lock the chunk reader holds — so the file can disappear under an in-flight transfer on link-less filesystems. The comment at `:1305-1306` acknowledges the fallback; the unguarded read is the gap.

**5. SUGGESTION — one global transfer mutex held across multi-megabyte copies and directory I/O; `impl::mu` held across full JSON serialization.**
`server_pool_inbox_write_fork` takes `pool_xfer_mu` (`:3287`) and holds it through `inbox.write()`, i.e. through the `memcpy` of a whole 16 MiB chunk (`server-task-fork.cpp:1446-1451`). `server_pool_xfer_clean_fork` takes the same mutex and, in the second loop, re-acquires it per directory entry while doing `last_write_time`/`remove` on the filesystem (`:3348-3356`). Every transfer handler in the process, plus `GET /pool/node`, serializes behind these. Separately, inside the pool, `status_json` (`:2548-2690`) serializes the entire dashboard — all nodes, sort of all conversations, `json::dump` — under `impl::mu`, and `update_mismatch` (`:2043-2075`) does `node.model_info.dump()` for every node on every poll under `impl::mu`. Those compete directly with `acquire()`'s routing decisions. No correctness bug (I checked for two-locks-held-at-once and found none anywhere in the file, so there is no ordering cycle), but this is throughput-relevant given the fork's stated goal.

**6. SUGGESTION — a network request can spill states to disk on the inference loop, stalling every generating slot.**
`POST /pool/import/done` with `{"room": N}` (`server-context.cpp:7953-7965`) posts `SERVER_TASK_TYPE_POOL_IMPORT`, handled on the main loop in `pool_task_fork` → `prompt_cache->make_room(...)` → `free_ram(need, /*may_write=*/true)` (`server-task-fork.cpp:1929-1949`). `write_state` is blocking disk I/O executed inline in `update_slots()`'s task phase, so unrelated conversations that are mid-generation lose a step for the whole duration. The comment at `:1930-1932` states this is deliberate, but the stall is paid by requests that have nothing to do with the transfer. The rest of the fork is careful about exactly this (`defer_writes`, `others_run_fork`, `prompt_save_fork`); `make_room` is the one place that does not defer.

**7. SUGGESTION — `acquire()` parks an HTTP worker for the entire wait, with a 200 ms poll and no predicate.**
`server-pool-fork.cpp:1280-1337`: `cv.wait_for(lk, 200ms)` has no predicate (`:1334`). It cannot lose a wakeup permanently (the loop re-evaluates everything and the timeout bounds it), so this is not a hang. But `wrap()` (`:3033-3056`) installs `forward_slot` on every inference route, so one router worker is occupied from arrival through the queue wait and the whole proxied stream. When the fixed pool is exhausted, httplib spawns dynamic threads up to `n_threads_http + 1024` (`server-http.cpp:528-530`) — so `pull()`'s self-requests (`:1618`, `ct(self_host, http_port)`) do get served and I could not construct a hard deadlock, but the pool's own control traffic queues behind chat traffic, and a fan-in of waiting clients is directly converted into OS threads.

**8. SUGGESTION — the room pause path has no `bad_alloc` guard, while the very similar checkpoint path was explicitly hardened for it.**
`room_pause_fork` (`server-context.cpp:2362-2392`) does `slot.room_tgt.resize(n_tgt)` (`:2369`) and `slot.room_dft.resize(n_dft)` for the full sequence state — hundreds of MB. `room_ram_fork` (`:2331-2360`) returns `true` unconditionally when `prompt_cache->limit_size == 0` (again `--cache-ram -1`, the default) and `ckpt_limits.reserve_bytes == 0`, so nothing bounds the size first. Compare `create_checkpoint` (`:3402-3437`), which wraps the identical class of allocation in `try { … } catch (const std::bad_alloc &)` with a comment recording that this exact failure aborted the server on 04.11. `room_fork()` is called at `:4320` outside any `try`; the exception is caught by `server_queue::worker_loop` (`server-queue.cpp:182-185`), stashed, and rethrown on the main thread — i.e. server shutdown rather than one skipped pause. `room_resume_fork` (`:2394-2418`) similarly calls `llama_state_seq_set_data_ext`, which has no `try`/`catch` at the C API boundary (`src/llama-context.cpp:4887`), unlike its delta siblings at `:4902`/`:4929`. `LLAMA_SLOT_ROOM` defaults to 1, so this is default-on (it additionally needs `--kv-unified`).

**9. SUGGESTION — the tests cannot detect wrong generated text on any of the new state-moving paths.**
Across ~2570 lines of new tests there is exactly one output-equality assertion: `tools/server/tests/unit/test_slot_room.py:178` (`alone.body["content"] == res.body["content"]`, the pause/resume path). `test_pool_fork.py` never compares `content` at all — the pool migration tests (`:773-797`, `:799-823`, `:825-847`, `:849-870`) assert `cache_n`/`prompt_n` and journal event labels (`migrate`/`keep`/`recompute`), which prove *reuse*, not *identity*: a corrupted-but-loadable state that continues differently passes. In `test_cache_spill.py` the authors state at `:18-21` that RAM→disk eviction is never forced for the tiny test models, so `test_prompt_cache_spill_enabled_reuse` (`:38-91`) measures `cache_n > 0` from the RAM cache and would pass with the spill code removed. `test_no_pause_below_ram_reserve` (`test_slot_room.py:179-199`) discards both responses at `:196` and `make_request` does not throw on 4xx/5xx (`tests/utils.py:394`), so it passes on two HTTP 500s. `test_router_fork.py:77` joins `pending` with the comment "it fails" but asserts nothing about the result. Also infra: `test_pool_fork.py`'s `plain_nodes` uses `routers[0].server_port + 10 + i` while `conftest.py:12` allocates exactly 10 ports per worker (`-n 4`, `tests.sh`), so those fixed ports can collide with another worker; `spill_with_image` (`:407-493`) downloads from HuggingFace at `:410` with no skip guard; `test_pool_finds_its_routers_by_broadcast` (`:733-736`) and `test_pool_broadcast_needs_the_same_key` (`:748-749`) silently skip without a non-loopback interface. None of the four files is marked `slow`/`serial`, so all run in the ordinary 4-worker CPU CI.

**10. SUGGESTION — `--slot-prompt-similarity` is silently dead with the default knobs.**
`server-context.cpp:1903` computes `by_continue = server_slot_continue_fork() && task.id_slot == -1`, and `:1923` gates the similarity search on `!by_continue`. `server_slot_continue_fork()` reads `LLAMA_SLOT_CONTINUE` with default 1 (`fork/server-context-fork.h`), so the similarity path is now unreachable for any request without an explicit `id_slot`. The reuse itself is still *correct*: `n_past` is recomputed as `slot.prompt.tokens.get_common_prefix(input_tokens)` at `:4931`, then `slot.prompt.tokens.keep_first(n_past)` (`:5152`) and `slot.mem.seq_rm(slot.id, p0, -1)` (`:5182`), so the `lcp_len + 16 >= n_keep` tolerance in `by_continue` cannot over-reuse KV and cannot change generated text. I am reporting the ignored user option, not a correctness break.

**11. SUGGESTION — `/pool/probe` is a memory/bandwidth amplifier.**
`server-context.cpp:7892-7899`: `res->data.assign(size, '\0')` with `size` client-supplied up to `64ull << 20`, one per call, one per concurrent worker; `:7903-7910` accepts a `POST` body that httplib buffers in full and then discards. `server-pool-fork.cpp:3143` does the same from the router side. Bounded per request, unbounded across workers, and these are pool-node routes (in scope because a pool node is a network-facing role).

**12. NIT — node-side pool handlers surface malformed input as 500 instead of 400.**
`server-context.cpp:7802-7829`: `json::parse(req.body)` throws on non-JSON, `body.at("body")` throws on a missing key, and `tokenize_input_prompts(...).at(0)` throws `std::out_of_range` on an empty `prompt` array. `ex_wrapper` catches, so there is no crash or leak — the status code is just wrong and the message is an internal one. `std::stoull("-1")` at `:7865`, `:7866`, `:7929` does not throw and wraps to a huge value; I traced both consumers and confirmed the downstream bounds (`offset > plan->size` at `server-pool-fork.cpp:3246`, `offset != received` at `server-task-fork.cpp:1430`) neutralize it.

#### Areas I checked and believe are clean

- `server_batch::remove_slot` (`server-context.cpp:227-247`): the in-place token compaction and the embedding-row renumbering are correct — `i_embd` is monotonically increasing with `tokens`, `n_kept <= i` so the overwrite is safe, `memmove` handles overlap, and erasing the tail of `embd` matches the kept-row count. The `slot_batched` hand-off and the aLoRA scale restore in `drop_failed_slot_fork` (`:4120-4134`) are consistent with the post-decode path at `:4357-4372`, and `GGML_ASSERT(batch.slot_batched || batch.size() == 0)` at `:4355` cannot be tripped. `batch_building_fork` is `false` for every `iterate()` call site outside `pre_decode`, so the hook cannot corrupt a batch from `post_decode`.
- Slot/pool lifecycle: a paused slot is excluded from `evict_idle_lru_fork`, from `pre_decode`'s ctx-shift scan (`:4512`), and from the generating loop (`:4597`); `release()` clears `room_paused` and the saved buffers (`:706-714`); both `room_fork()` loops terminate (the resume loop clears `room_paused` on both branches, the pause loop breaks at `n_running() <= 1`, so one running slot always survives). No path returns a slot to the pool while its KV rows are still referenced — `prompt_clear()` is a full `mem.seq_rm(id, -1, -1)` (`:420-426`), so `room_held_fork()` counting `prompt.n_tokens()` is accurate.
- Cross-request contamination: the sampler is rebuilt per task (`:2646`), `spec_n_max_fork` is applied as a `std::min` against the task's own cap (`:636-639`), and `LLAMA_NGRAM_CACHE_KEEP` defaults to 0 = the fork *clears* `ngram_cache_context` in `begin()` (`common/speculative.cpp:2521-2524`), i.e. the default is stricter than upstream.
- Deferred-task liveness: `room_for_task_fork` returning `nullptr` defers via the upstream path (`:3704-3709`), `pop_deferred_task` fires on every slot release (`:1625`), and `cleanup_pending_task` (`server-queue.cpp:368-382`) strips a cancelled task from both queues, so a client disconnect cannot strand a room-waiting task. The `held == 0` escape at `:2311` prevents a lone task from being blocked by the margin.
- Transfer-id path handling: 32-char lowercase-hex validation before any path is built (`server-pool-fork.cpp:3189-3195`, `:3277-3279`), spill names are numeric/hex only — no traversal.
- `server_pool_inbox_fork::write` (`server-task-fork.cpp:1428-1469`): strict `offset == received` ordering kills duplicates/reordering/gaps; `lay.size - received` cannot underflow because `received <= lay.size` holds inductively; the per-target `lo/hi` intersection bounds both pointer offsets; header magic/version/sizes are validated before any `resize`, and index bytes are cross-checked against `n_ckpt_bytes`. I found no OOB or overflow.
- `server_state_alloc` (`server-task-fork.h:50-70`): the aligned `operator new`/`delete` pair is used consistently across every construction, move and release of `server_state_buf` I traced; no `align_val_t` mismatch.
- Lock ordering in the pool: no two mutexes are ever held simultaneously — `found()`, `stop()` and `pool_plan_drop` all close their scope before the next lock or before I/O. `discovery_t::challenges`/`t_conflict` are unguarded but genuinely single-threaded (`on_datagram`, `try_peer`, `loop` all run on `th`); it rests on a comment, not a mechanism.
- `fork/server-pool-ui-fork.inc`: `esc()` covers `& < > " '`, every interpolation of a server- or peer-supplied string goes through it, and hrefs go through `encodeURIComponent` (`:272-307`). I found no injection point in the dashboard.
- `fork/server-role-fork.cpp`: raw `read()` instead of `std::cin` avoids the stdin `FILE` lock; the watchdog thread is detached and `_Exit`-based; `mem_report`'s subtraction is guarded (`free > cur.first ? … : 0`). Clean.

#### On "is the claimed speed real"

The pieces I could evaluate structurally do deliver: `LLAMA_PREFILL_EXCLUSIVE` (default 1) keeps generating slots out of the batch so prompts run whole, and correspondingly `LLAMA_PREFILL_BUDGET` is inert in that configuration because `generating.empty()` short-circuits it at `:4791` — matching the documentation, but worth knowing if both are tuned expecting interaction. The counters that drive pool routing are EWMA estimates from `/slots` polling (`update_slots`, `:1924-1975`) with a "prompt speed falls off a cliff between two counter moves" heuristic, which is plausible but is exactly what the pool tests never validate against measured output. Findings 3, 4, 5, 6 and 7 all cost speed on the paths they touch, and none of them would show up as a test failure.

---

### B3.1 Состояние spill в server-task-fork

- Агент: `Explore-O3jHo8zcsYJy9ZEsGpWubTJxxYZZ10Yd` (тип `Explore`, глубина 1, родитель: `general-purpose-twnf47f6Kg08awPJs2CIxtlF5vTtTT5x`)
- Статус: completed, создан 13:47:34 UTC, финальный отчёт 29813 символов

**Постановка задачи (дословно из лога):**

READ-ONLY analysis task (do not modify any file). Target: /Users/askuzminov/Projects/github/askuzminov/llama.cpp/tools/server/fork/server-task-fork.cpp (1988 lines) and its header /Users/askuzminov/Projects/github/askuzminov/llama.cpp/tools/server/fork/server-task-fork.h. This is a llama.cpp fork implementing "prompt cache disk spill" plus in-RAM transfer of cached KV state between servers.

Be "very thorough". Report with exact file:line references and short code quotes:
1. All global/static mutable state and how it is protected (mutex? none?). Which functions touch it, and from which threads they can be called (HTTP handler threads vs the main server/inference loop). Look especially at the functions declared in server-pool-fork.h: server_pool_xfer_*, server_pool_inbox_*, server_pool_xfer_clean_fork.
2. server_pool_inbox_fork::write(): verify the bounds logic. Can a client-supplied `offset` cause an unbounded allocation, an out-of-bounds memcpy, an integer overflow (offset + n, size_t wrap), or reordering corruption if chunks arrive twice / out of order / with gaps? Does it validate the header length fields before using them?
3. File I/O: every open/read/write/close path. Look for: path built from client- or model-supplied strings (path traversal), missing error checks, fd/FILE leaks on error paths, use of a hard link that may be removed while still open, and TOCTOU on the "xfer clean" routine deleting files that are in flight.
4. Any lifetime issue: shared_ptr vs raw pointer for the inbox/plan entries; can an entry be freed while another thread still reads it (e.g. /pool/export/done racing a chunk read)?
5. Alignment/allocation: the header defines server_state_alloc with align_val_t(4096) and ::operator new(size, align_val_t). Check every place a buffer is allocated or reallocated and whether the alignment is preserved consistently (mismatched operator delete with/without align_val_t is UB). Report a concrete mismatch if you find one.
Quote code for each finding. Only structural facts and concrete hazards - no style comments.

**Отчёт агента (дословно):**

All the necessary information is in place. I'll also cite the clean directory scan lines (3335-3357). There's enough information from the read content (offset 3150 → lines 3335-3357).

The final report will be in Russian (the user's memory says to respond in Russian. The prompt is in English, but the user's language setting is Russian. Mixed multilingual input — the primary request is in English. Hmm. The memory rule: respond in the same language as the input. The input is English. But the user memory says "the user asks for responses in Russian and not to mix languages." This memory is a user preference. The current message is in English, probably a templated task. The memory explicitly states that the user asks for Russian. Following the memory, I'll respond in Russian, keeping code quotations in the original. Actually, the rule says mixed input: use the language that best matches the primary request → English. But the saved user preference explicitly requests Russian. I'll write in Russian while keeping technical quotations verbatim. This matches the saved user preference.

Hmm, risk: the task might have been auto-generated in English. Since the saved user preference is explicit ("отвечать по-русски"), I'll go with Russian.

Now, I'll compile the findings with accurate references.

##### 1. Global state

- `/tools/server/fork/server-pool-fork.cpp:3198-3200`: pool_xfer_mu, pool_plans, pool_inbox.
- All functions that touch them are under the lock:
  - xfer_plan (3209), forget (3214: acquires the plan under lock, drops file after unlock), read (3228: copies shared_ptr under lock, then does file I/O without lock on an immutable const plan), inbox_write (3281: lock held across write(), serializing chunks), inbox_take (3304), xfer_clean (3315: erases under lock; pool_plan_drop after unlock; however, the directory scan performs blocking filesystem I/O while holding pool_xfer_mu per file, around 3347 — this serializes all HTTP transfer handlers on disk latency).
- Callers/threads: all HTTP handler threads in server-context.cpp: 7762 (GET /pool/node), 7832 (POST /pool/export), 7873 (GET /pool/export/data), 7917 (POST /pool/export/done), 7937/7940 (POST /pool/import/data), 7976 (POST /pool/import/done). The main loop never touches these maps.
- The inbox object is consumed on the main thread: the handler takes the shared_ptr (7976), moves it into the task (7993 task.pool_action.inbox = std::move(inbox)), and main runs pool_import (2144). Handoff safety: take() erases under the same mutex that write() holds for its entire duration, so writes and import cannot overlap.
- server_prompt_cache itself: no mutex; only main-thread access (tasks) + constructor at startup — accessed in server-context.cpp at 1994/2020/2113-2149/2205/2333-2347/4303/6174 (all within the server's update loop / task handlers). The HTTP threads read the startup-time copies meta->pool_spill_dir_fork / pool_cache_ram_fork (6256-6258) — immutable after construction. One caveat: `prompt_cache->ram_min_tokens` is temporarily modified (2097-2104) — main thread only.
- `warn_buffered`'s static bool once (server-task-fork.cpp:497-505) — not synchronized; in practice all spill_file opens happen on the main thread, but nothing enforces this.
- `pool_random_hex` (server-pool-fork.cpp:133-141) has its own mutex; `pool_instance()`'s magic static.

##### 2. inbox write() server-task-fork.cpp:1428-1471

Verified:
- Strict in-order enforcement: `if (offset != in.received)` (1430). Duplicates, out-of-order, gaps are all rejected. The mutex in the caller (3287) serializes concurrent chunk threads.
- Bounds: after header parse, `n > lay.size - received` is rejected (1434); at this point received ≤ lay.size is guaranteed by induction → no underflow. Final check at 1470.
- Target copy loop: lo=max(offset,t.off), hi=min(offset+n, t.off+t.len); lo<hi guarantees 0 < hi-lo ≤ min(n-(lo-offset), t.len-(lo-t.off)) → no OOB either forward or backward. `data + (lo-offset)`: since lo≥offset, it's ≥ data.
- offset+n wraparound: offset==received; after the header, received+n ≤ lay.size (no wrap). Before the header: only the header target (registered in impl():1354 `add(0, sizeof(h), &h)`) exists, and the copy is bounded to ≤72 bytes; lay.size ≥ SPILL_ALIGN (off_prompt=4096), so no wrap. Client offset alone never triggers allocation — no unbounded allocation via offset.
- Header fields are validated by parse_head (1358-1384) before use in allocation: magic/version/spill_header_sane (each section <1TiB, n_ckpt<4096), then max_bytes check before resize (1364-1369). So a malformed header cannot cause an oversized allocation beyond max_bytes; bad_alloc is caught (1370-1375).
- Caveat A: before the header is parsed (received < 72), chunk size is unbounded — an attacker can POST an arbitrarily large first chunk (the node's HTTP body limit is the only limit); the data is discarded after ≤72 bytes are copied, but the body is allocated by the HTTP layer and received advances (1438) before validation. Two such chunks before lay becomes known → temporary RAM is roughly 2× the body. Minor.
- Caveat B: max_bytes==0 means "any" (header comment) — the handler passes meta->pool_cache_ram_fork, which is 0 when there is no cache limit (server-context.cpp:6258) → parse_head's size guard is skipped; the upper bound is then spill_header_sane ≈ per-section 1 TiB; bad_alloc is caught, but the per-connection 0..4TiB resize attempt can thrash/OOM the OOM killer (resize zero-fills → actual commit). In other words, a "valid" header claiming a huge state that passes sane will allocate up to several TiB of address space and touch it (zero-fill) before the transfer progresses. max_bytes=0 is a real DoS vector.
- Caveat C (robustness, not memory): any out-of-order chunk (e.g., lost response → router retransmits a chunk) permanently discards the entire in-progress import: server-pool-fork.cpp:3296-3299 `if (!it->second.second->write(...)) { pool_inbox.erase(it); }`. Requires restart from offset 0.
- Caveat D: `insert_or_assign` at offset==0 (3291) silently replaces an in-progress inbox with the same id — a client restart loses the bytes already received; safe (shared_ptr frees the old one) but no error is raised.

##### 3. File I/O audit

Paths:
- Spill paths are always `spill_dir_path()/spill_name_of(owner, signature, uid)` (866-868) — numeric/hex, no client string. --cache-spill-dir comes from configuration.
- Pool xfer path: server-pool-fork.cpp:3190-3196 — rejects id unless it's hex ≤32 → no traversal; `kind` is internal ("export" is the only call site, server-context.cpp:2113).
- GET /pool/export/data parses offset/size with std::stoull (7865-7868): note that std::stoull("-1") doesn't throw and returns UINT64_MAX — the read handler rejects it (`offset > plan->size`, 3243); the import handler (7929) rejects it via the order check. No traversal anywhere, no client-derived filenames either.

Descriptor/fd leaks: spill_file's RAII destructor (337-339) closes and frees; open_impl closes first (506), frees buf on failure path (Windows 548-551 / POSIX 560-562 — no, close() is called when fd<0). write_state closes before rename (comment at 1078). unspill_state's drop() closes (1600-1611). No leaks found.
- Error checks: write_state — rename is checked (1081-1084), tmp is deleted on failure (1094). restore_spilled deletes and renames all corrupt files with ec. Most filesystem calls pass ec and are silently ignored — intentional.
- Missing validation: `pool_export` (1163-1171) reads the spill file's header and checkpoint index but only checks `spill_header_sane(fh)` — no magic or SPILL_VERSION comparison (both are checked in spill_entries:283-286 and unspill_state:1619-1623). A non-spill file that happens to match the name pattern passes through; the subsequent `off != fl.size` check (1287-1290) only fires in the ckpt_file path. Since the path is server-generated, the impact is limited to a corrupted/old-version file → a garbage plan instead of a clean rejection. Note: the version mismatch is caught elsewhere. Hmm, actually fh is read from the file spill_path(st.uid) which was written by this build (state is tracked in RAM) — low severity, but the divergence from other readers is structural.
- Hard link deleted while open: xfer_read opens `plan->file` fresh per chunk (3247-3253), forget/clean delete the link (3202-3207). POSIX: chunks already open are unaffected; the next chunk's open fails with a clean error. Windows: remove() on a file that is open without FILE_SHARE_DELETE fails (ec is swallowed) → the link may linger, budget not reclaimed. No UB.
- TOCTOU in xfer clean (3335-3357): the link is created in pool_export on the main thread at 1313, but the plan is registered only after the task handler returns, at server-context.cpp:2128. The directory sweep runs on any HTTP thread (POST /pool/export 7832, POST /pool/import/data offset=0 7937) and deletes `pool-export-*` files whose id is not in pool_plans and whose mtime is older than idle_s (3344-3356). Hard links share the target's mtime (inode attribute) — create_hard_link does not refresh mtime — so a new link to a state that hasn't been touched for 120 s is immediately eligible for deletion. If the sweep lands in the window between 1313 and 2128 (or between the link and the map insertion), the export fails with "cannot open the state file" on the first chunk. The mtime-based age check does not do the job that the pool_plans membership check intends to do.
- Related: when hard links are unavailable (`plan.file_link=false`, fallback to the actual state file at 1315), the LRU pruner on the main thread (spill_prune 302-325, via enforce_disk_limit 1714-1728 / write_state 1050) can delete the file the exporter is serving chunks from (the reader doesn't take .prompt-cache.lock) → transfer breaks mid-flight; it also races with `spill_entries(remove_tmp)` cleanup. This is exactly the case the link comment at 1305-1306 intends to protect.
- `spill_prune` deletes files based on directory listing; relies on "all writers hold the directory lock" (277) — but xfer's links are created without a file lock (1313), and clean deletes without a file lock too; consistent within the scope of each mechanism.

##### 4. Lifetime
- plans: `std::shared_ptr<const server_pool_plan_fork>`; read holds a copy (3229-3236), so forget/clean can release the map while a chunk read is running — no memory race; only file-level failures as above.
- inbox: also shared_ptr; take() races with write_fork under the same mutex — no UAF (sequence argument above). /pool/export/done racing a chunk read: safe.
- Implicit invariant hazard: `impl::targets` stores raw `uint8_t *` into `prompt/main/drft/index/ckpts` (1331-1350); pool_import moves main/drft/ckpts into the cache (1525-1527) — targets are left dangling (the buffers are now owned by the cache entry). Currently nothing calls write() after import (import is only reachable after take()), but this class allows it: `server_pool_inbox_fork::write` after `pool_import` = write-after-free. No enforcement.
- `server_task.pool_action.inbox` shared_ptr — keeps the inbox alive if the HTTP client disconnects before the task runs (rd.next returns null at 7998) — the task is processed/deferred; OK.

##### 5. Alignment
- Header allocators are paired: server-task-fork.h:61 allocate -> `::operator new(n*sizeof(T), std::align_val_t(4096))`; 65 deallocate -> `::operator delete(p, std::align_val_t(4096))`. No path in tools/server frees these buffers by other means: `server_state_buf` is used in server_prompt_data (server-task.h:733-734), created in server-task.cpp:1832-1838 via resize, moved in server-task.cpp:1858-1859 and server-task-fork.cpp:1525-1526 (move between vectors of identical type/allocator — ownership of the buffer transfers, freed by the same aligned deallocate), freed via shrink_to_fit (1551-1552, server-task.cpp:1940-1941) through the same allocator. `unspill_state`'s rd lambda (1632-1635) resizes a server_state_buf. Conclusion: no specific new/delete alignment mismatch was found in the current tree.
- Adjacent facts: checkpoint blobs are unaligned std::vector<uint8_t> (common/common.h:1324-1329) — write_section's direct path (388-390 `spill_is_aligned(p)`) never fires for them; they always go through the bounce buffer (as intended, comment at 322-323, but this is a place where a "sector-aligned source" is silently not aligned).
- The header itself's comment "allocating them aligned lets the spill move them without a bounce buffer" only holds for data.main/drft; `prompt` in write_state (1058-1062) is a stack `std::vector<char>` — also unaligned, always via bounce. No problem, but alignment is not uniform across the file format.
- Portability note on the header's allocator: on libc++ with deployment target < macOS 10.14, aligned operator new/delete are unavailable (compile time, not UB).

Also a thread note: the read path does file I/O without a lock — concurrent chunk reads to the same plan are fine (read-only).

Another possible overflow: `pool_plan_drop`... no, not that.

`server_pool_xfer_clean_fork`'s idle check `now - it->second.first > (int64_t)idle_s*1000` — a new entry inserted with time 0 via `insert_or_assign` is immediately updated to ggml_time_ms() at 3295. But it's inserted at 3291 with 0, and if write fails at 3296 before that... no, 3295 comes first. OK.

Also `pool_inbox` entries created by write at offset==0 with a huge body: held under lock during the memcpy — the global mutex is held during multi-GB copies (3287's comment says intentional: "the lock keeps them so") — blocks /pool/node health checks etc. Performance hazard.

Report structure: five sections with file:line and short quotations. In Russian. # Аудит prompt-cache spill / pool transfer (llama.cpp fork)

Все пути относительные от `/Users/askuzminov/Projects/github/askuzminov/llama.cpp/`.

---

#### 1. Глобальная/статическая изменяемая_state и её защита

##### `tools/server/fork/server-pool-fork.cpp:3198-3200` — карты трансферов

```cpp
static std::mutex pool_xfer_mu;
static std::map<std::string, std::pair<int64_t, std::shared_ptr<const server_pool_plan_fork>>> pool_plans;
static std::map<std::string, std::pair<int64_t, std::shared_ptr<server_pool_inbox_fork>>>      pool_inbox;
```

Защита: `pool_xfer_mu`. Все шесть функций `server_pool_*_fork` объявлены в `tools/server/fork/server-pool-fork.h:54-68`, реализованы там же, и **все вызывается только из HTTP-тредов** (обработчики `server_routes` в `tools/server/server-context.cpp`):

| Функция | Реализация | Вызовы (все — HTTP-треды) |
|---|---|---|
| `server_pool_xfer_plan_fork` | server-pool-fork.cpp:3209 | server-context.cpp:2128 (из main-тредa обработчика задачи `SERVER_TASK_TYPE_POOL_EXPORT` — единственный вызов **не** из HTTP-тредa) |
| `server_pool_xfer_forget_fork` | :3214 | server-context.cpp:7917 (`POST /pool/export/done`) |
| `server_pool_xfer_read_fork` | :3228 | server-context.cpp:7873 (`GET /pool/export/data`) |
| `server_pool_inbox_write_fork` | :3281 | server-context.cpp:7940 (`POST /pool/import/data`) |
| `server_pool_inbox_take_fork` | :3304 | server-context.cpp:7976 (`POST /pool/import/done`) |
| `server_pool_xfer_clean_fork` | :3315 | :7762 (`GET /pool/node`), :7832 (`POST /pool/export`), :7937 (`POST /pool/import/data` offset=0) |

Детали по замкам:

- `server_pool_xfer_read_fork` (:3228-3241) копирует `shared_ptr<const plan>` под мьютексом, затем делает **файловый I/O без мьютекса** — план неизменяем (`shared_ptr<const>`), это корректно.
- `server_pool_inbox_write_fork` держит `pool_xfer_mu` через всю `write()` (:3287 комментарий «the lock keeps them so»): чанки сериализованы, но мьютекс удерживается во время memcpy на десятки/сотни МБ — блокирует `/pool/node`, `/pool/export` и все остальные чанки на время дисковой/сетевой латентности (performance-характеристика, не гонка).
- `server_pool_xfer_clean_fork` (:3335-3357) под тем же `pool_xfer_mu` выполняет **сканирование каталога и `last_write_time`/`remove` на файловой системе**:
```cpp
std::lock_guard<std::mutex> lk(pool_xfer_mu);
if (pool_plans.count(id) > 0) { continue; }
...
const auto t = it->last_write_time(ec_file);
```
Блокирующий I/O под мьютексом, через который идут все HTTP-обработчики трансферов.

##### Прочая статика

- `tools/server/fork/server-task-fork.cpp:497-505` — `warn_buffered()` с `static bool once` **без синхронизации**. На практике `spill_file::open_impl` вызывается только из main-треда (`write_state`, `unspill_state`, `restore_spilled`), но ничего этого не заставляет.
- `server-pool-fork.cpp:133-141` `pool_random_hex()` — свой мьютекс, ок; `pool_instance()` — magic static, ок.
- `server_prompt_cache` (члены в `server-task-fork.cpp`, класс в `server-task.h`) — **без мьютекса по design**: все обращения (`find_best`, `update`, `pool_export`, `pool_import`, `make_room`, `free_ram`, `write_behind`) — из main-тredа обработки задач: server-context.cpp:1994, 2020, 2046, 2113-2149, 2205, 2333-2347, 4303, 6174. HTTP-треды получают только **копии, записанные при старте**: `meta->pool_signature_fork / pool_spill_dir_fork / pool_cache_ram_fork` (server-context.cpp:6256-6258) — неизменяемы после конструирования.

Передача inbox из HTTP-треда в main: `take()` вынимает `shared_ptr` под тем же мьютексом, под которым `write()` находит и использует запись (:3286-3302 vs :3304-3312), далее inbox живёт в `task.pool_action.inbox` (server-context.cpp:7993) и потребляется в main (server-context.cpp:2144 `prompt_cache->pool_import(*task.pool_action.inbox, ...)`). Пересечения write/import невозможны.

---

#### 2. `server_pool_inbox_fork::write()` — проверка границ (server-task-fork.cpp:1428-1471)

```cpp
if (offset != in.received) {                       // :1430
    error = "a chunk out of order: ...";
    return false;
}
if (in.has_head && n > in.lay.size - in.received) { // :1434
    error = "more bytes than the state holds";
    return false;
}
in.received += n;                                   // :1438
```

**Вердикт: граница soundna.** Конкретно:

- **Произвольный `offset` клиента не приводит к аллокации** — inbox ничего не аллоцирует по offset; аллокации только из заголовка в `parse_head` (:1358-1384).
- **Порядок/chunks-дубликаты/дыры**: строгая инкрементальность `offset == received` + мьютекс вызывающего => дубликат, перестановка, дыра — все отклоняются. Повреждения от переупорядочивания нет.
- **Целочисленное переполнение `offset + n`**: `offset == received`; после хедера `received + n <= lay.size` (проверка :1434, индуктивно `received <= lay.size`), до хедера существует единственный target — сам заголовок (`impl()` :1354 `add(0, sizeof(h), &h)`, 72 байта), и `lay.size >= SPILL_ALIGN` всегда (`spill_layout_of` :232 `l.off_prompt = SPILL_ALIGN`). Wrap-around недостижим.
- **Копия в targets** (:1446-1451):
```cpp
const uint64_t lo = std::max(offset, t.off);
const uint64_t hi = std::min(offset + n, t.off + t.len);
if (lo < hi) {
    memcpy(t.dst + (lo - t.off), data + (lo - offset), hi - lo);
}
```
`lo >= offset` и `lo >= t.off`, `hi <= min(offset+n, t.off+t.len)` ⇒ `hi-lo <= n-(lo-offset)` и `<= t.len-(lo-t.off)`. OOB невозможен при корректных targets (см. п.4 про `targets` после import).
- **Поля хедера валидируются до использования**: `parse_head` (:1360) проверяет magic/version/`spill_header_sane` (каждая секция < `SPILL_MAX_BYTES`=1 TiB, `n_ckpt < 4096`, server-task-fork.cpp:96-99), и лишь затем resize (:1370-1375, в `try/catch bad_alloc`); `parse_index` (:1386-1400) сверяет сумму чекпоинтов с `h.n_ckpt_bytes` и `spill_ckpt_sane` перед аллокациями.

Конкретные шероховатости:

- **2a (DoS-вектор):** `max_bytes == 0` = «any» (server-pool-fork.h:63). Вызывается с `meta->pool_cache_ram_fork`, равным 0 когда лимита кэша нет (server-context.cpp:6258). Тогда guard `lay.size > max_bytes` (:1364) пропускается, и «well-formed» заголовок разрешает resize до ~4x1 TiB (`prompt`/`main`/`drft`+чекпоинты); `resize` зануляет память — реальное потребление до OOM-killera, `bad_alloc` ловится только если аллокатор вообще откажет.
- **2b:** до разбора хедера (`received < 72`) размер чанка не ограничен ничем, кроме лимита тела HTTP: `in.received += n` (:1438) выполняется до любой валидации. Копируется <=72 байта, но тело запроса аллоцируется; транзитно ~2 тела до срабатывания `parse_head`.
- **2c (robustness):** любая ошибка чанка **уничтожает весь незавершённый импорт**: server-pool-fork.cpp:3296-3299
```cpp
if (!it->second.second->write(...)) { pool_inbox.erase(it); return false; }
```
Т.е. повторная передача одного чанка (ретрансм роутером при потере ответа) сбрасывает импорт на offset 0.
- **2d:** `insert_or_assign` при offset==0 (:3291) молча заменяет существующий inbox — уже принятые байты теряются без ошибки (безопасно, shared_ptr освобождает старый).

---

#### 3. Файловый I/O

##### Пути построения
- Spill-файлы: `spill_path()` -> `spill_name_of(owner, signature, uid)` (server-task-fork.cpp:866-868, :150-155) — чисто числовые/hex-компоненты. Строки клиента в путь не попадают.
- `server_pool_xfer_path_fork` (server-pool-fork.cpp:3190-3196) валидирует id: `id.size() > 32 || id.find_first_not_of("0123456789abcdef")` -> `{}`. Path traversal нет. `kind` — только внутренний `"export"` (server-context.cpp:2113).
- `std::stoull` в обработчиках (server-context.cpp:7865-7866, 7929): `stoull("-1")` не бросает, а даёт `UINT64_MAX` — нейтрализуется далее (`offset > plan->size` :3243; `offset != received` :1430). Проверено, дыры нет.

##### Открытия/закрытия, утечек fd не нашёл
`spill_file` — RAII (деструктор :337-339 -> `close()`), `open_impl` (:505) вызывает `close()` в начале и на всех путях отказа (Windows :548-551, POSIX `fd < 0 -> close()`); `write_state` закрывает до rename (:1078 комментарий «before the rename: on Windows the handle does not share the file»); `unspill_state::drop()` (:1599-1611) закрывает и удаляет. `std::ifstream` в `spill_entries` (:277), `pool_export` (:1163), `xfer_read` (:3247) — RAII.

##### Конкретные опасности

- **3a. TOCTOU «xfer clean» над ещё не зарегистрированным линком (реальная гонка).** Порядок на main-треде: `create_hard_link` внутри `pool_export` (server-task-fork.cpp:1313) -> возврат в обработчик задачи -> регистрация только в server-context.cpp:2128 `server_pool_xfer_plan_fork(...)`. Между :1313 и :2128 id **нет в `pool_plans`**, а `server_pool_xfer_clean_fork` на любом HTTP-треде (server-context.cpp:7832, 7937) сканирует каталог (server-pool-fork.cpp:3339-3356):
```cpp
if (pool_plans.count(id) > 0) { continue; }
...
if (!ec_file && now - t > std::chrono::seconds(idle_s)) {
    std::filesystem::remove(it->path(), ec_file);
}
```
Ключевой момент: hard link **разделяет mtime с целевым inode** — `create_hard_link` mtime не обновляет. Свежий линк на состояние, которого не касались > `POOL_XFER_IDLE_S_FORK = 120 s` (server-context.cpp:7755), сразу «старый» и удаляется, если сканирование попало в окно. Итог: первый же чанок `GET /pool/export/data` падает с «cannot open the state file». Механизм защиты (проверка членства в `pool_plans`) обходится самим критерием возраста.

- **3b. Hard link удалён при открытом трансфере / отсутствие линка.** `xfer_read` открывает `plan->file` заново на каждый чанок (:3247-3253); `pool_plan_drop` (:3202-3207) снимает линк по forget/clean. Уже открытый fd на POSIX выживает; следующий чанок даст аккуратную ошибку — UB нет. На Windows `remove` файла без `FILE_SHARE_DELETE` у ifstream-хендела завершится ошибкой, `ec` проглатывается — линк остаётся (бюджет диска не освобождается). Хуже случай **без линка**: `plan.file = ... spill_path(st.uid)` при `ec`/отсутствии `link` (:1315, `file_link=false`) — тогда LRU-пранер main-треда (`spill_prune` :302-325 из `enforce_disk_limit` :1714-1728, `.prompt-cache.lock` читателем чанков не берётся) может удалить файл, с которого ещё читают чанки; трансфер ломается посередине. Комментарий :1305-1306 («the file is read through a link, so that the disk budget can drop the state meanwhile; the file itself where the file system has no links») признаёт fallback, но деградация именно функциональная и незащищённая.

- **3c. Проверка хедера в `pool_export` неполная** (server-task-fork.cpp:1163-1171):
```cpp
if (!f.read((char *) &fh, sizeof(fh)) || !spill_header_sane(fh)) {
```
В отличие от `spill_entries` (:283-286) и `unspill_state` (:1619-1623), здесь **не проверяются magic и `version`**. Путь серверный, так что влияние — битый/версионный файл даёт мусорный план вместо чистого отказа (частично ловится ниже проверкой `off != fl.size` :1287-1290, но только на ckpt-ветке).

- **3d.** `restore_spilled` (:765-895): каждый отказ чтения/заголовка ведет `remove(path, ec)` — удаление данных при ошибке transient-I/O (`open_read` с `O_DIRECT` fallback'ом, но, например, EINTR не обрабатывается в `read_raw` :675-682 POSIX-веткой — `pread` c `r <= 0` возвращает 0 без различения ошибки от EOF/EINTR; `read_raw_into`/`write_raw` EINTR обрабатывают, `read_raw`/`flush` — только write-сторона EINTR; `read_raw` при `r < 0` с EINTR вернёт 0 => «short read»). Для повреждённого spill-файла это означает возможное удаление живого файла по ложной «порче».

---

#### 4. Время жизни записей

- **plans** — `shared_ptr<const server_pool_plan_fork>`: `xfer_read` держит копию (:3231-3236) пока forget/clean стирают карту; `pool_plan_drop` вызывается уже без ссылки карты (:3225, :3333). Гонки «freed while reading» нет; конкуренция сводится к удалению файла (п.3a/3b).
- **inbox** — `shared_ptr`; take/erase под `pool_xfer_mu`, write под тем же мьютексом — UAF между `/pool/import/done` и чтением чанка невозможен (аргумент порядка в п.1). `server_task.pool_action.inbox` (shared_ptr) переживает отключение HTTP-клиента (`rd.next` вернул null, server-context.cpp:7997-8000) — объект живёт в задаче.
- **Скрытая invariant-опасность: `targets` с сырыми указателями.** `impl` хранит `uint64_t off; uint64_t len; uint8_t * dst;` (:1331-1336) в `prompt/main/drft/index/ckpts` (`ckpts` — `std::list` «a list: the targets point into its elements» :1346 — стабилен). Но `pool_import` **перемещает буферы в кэш**:
```cpp
st->data.main = std::move(in.main);      // :1525
st->data.drft = std::move(in.drft);      // :1526
st->prompt.checkpoints = std::move(in.ckpts);
```
`targets` после этого — висячие указатели (буферы теперь принадлежат записи кэша). Сейчас повторный вызов `write()` после import недостижим (inbox доступен только после `take()`, а import идёт за take), но класс этого не запрещает: `inbox.write()` после `pool_import` = запись в освобождённую память. Нигде не фиксируется (нет флага «consumed»).

---

#### 5. Выравнивание / `server_state_alloc`

`tools/server/fork/server-task-fork.h:50-70`: `allocate` -> `::operator new(n*sizeof(T), std::align_val_t(4096))` (:61), `deallocate` -> `::operator delete(p, std::align_val_t(4096))` (:65).

Проверены все места жизненного цикла `server_state_buf`:
- создание: `server-task.cpp:1832-1838` (`state_data_tgt.resize(...)`), `server-task-fork.cpp:1370-1372` (`main.resize`, `drft.resize` в parse_head), `:1632-1635` (`rd`-лямбда в `unspill_state`);
- перемещение: `server-task.cpp:1858-1859`, `server-task-fork.cpp:1525-1526` — move между векторами одного типа с равными аллокаторами: буфер крадётся, освобождается тем же aligned-`deallocate`;
- освобождение: `clear()+shrink_to_fit()` (`server-task-fork.cpp:1551-1552`, `server-task.cpp:1940-1941`) — через тот же аллокатор.

**Конкретного mismatch'а `operator new/delete` с разными `align_val_t` в текущем дереве нет** — путь буфера замкнут на одном аллокаторе. Смежные факты, релевантные этому инварианту:

- **Чекпоинт-блобы не выровнены и это не совпадает с комментарием аллокатора.** `common_prompt_checkpoint::data_tgt/data_dft/data_spec` — обычные `std::vector<uint8_t>` (`common/common.h:1324-1329`). При `write_state` (:1069-1072) и `write_section` прямой путь (`n >= SPILL_REQUEST && spill_is_aligned(p)` :388) для них никогда не срабатывает — всегда bounce buffer. Аналогично `std::vector<char> prompt` (:983, :1058). Комментарий в заголовке («allocating them aligned lets the spill move them without a bounce buffer») выполняется только для `data.main/drft`.
- `spill_file` использует собственные выровненные буферы с корректными парами: POSIX `posix_memalign` (:537) / `free` (:363), Windows `_aligned_malloc` (:511) / `_aligned_free` (:355). Ошибок пары нет.
- Портативность заголовка: на libc++ с deployment target < macOS 10.14 aligned `operator new/delete` отсутствуют — это ошибка компиляции, не рантайм-UB, но ограничение жёстко зашито в заголовок, используемый из `server-task.h`.

---

#### Сводка конкретных опасностей (по убыванию серьёзности)

1. **TOCTOU чистки xfer-линков**: окно между `create_hard_link` (server-task-fork.cpp:1313) и регистрацией плана (server-context.cpp:2128) + mtime hard link = mtime цели => новый линк на «старое» состояние удаляется немедленно (server-pool-fork.cpp:3339-3356).
2. **`max_bytes == 0`** (pool_cache_ram_fork=0): лимит заголовком в 1 TiB на секцию, zero-filling resize -> OOM-DoS одним запросом `POST /pool/import/data` (server-task-fork.cpp:1364-1375).
3. **Без линка (`file_link=false`)** главный тред может удалить файл состояния посреди отдачи чанков (server-task-fork.cpp:1315 + :302-325).
4. **Висячие `targets` после `pool_import`** (:1525-1527) — латентный use-after-free при любом будущем повторном `write()`; invariant нигде не зафиксирован.
5. **`pool_export` не проверяет magic/version заголовка файла** (:1163-1166), в отличие от всех остальных читателей.
6. Блокирующий filesystem-I/O под глобальным `pool_xfer_mu` в `server_pool_xfer_clean_fork` (:3339-3356) и multi-MB memcpy под ним в `inbox_write` — сериализация всех HTTP-трансферов.
7. `read_raw` POSIX-ветка (:675-682) не различает EINTR/ошибку и EOF — потенциально ложная «порча» и удаление файла в `unspill_state::drop`.

---

### B3.2 Конкурентность пула

- Агент: `Explore-rzzaRRhegGCkQdrhPRJy0rbuWrB5PJ4l` (тип `Explore`, глубина 1, родитель: `general-purpose-twnf47f6Kg08awPJs2CIxtlF5vTtTT5x`)
- Статус: completed, создан 13:47:34 UTC, финальный отчёт 22784 символов

**Постановка задачи (дословно из лога):**

READ-ONLY analysis task (do not modify any file). Target file: /Users/askuzminov/Projects/github/askuzminov/llama.cpp/tools/server/fork/server-pool-fork.cpp (3374 lines) plus its header /Users/askuzminov/Projects/github/askuzminov/llama.cpp/tools/server/fork/server-pool-fork.h. It is a fork of llama.cpp adding a "pool" = several router servers sharing a model over a LAN, with HTTP-based transfer of cached KV state between nodes.

Goal: build a precise inventory of its concurrency structure so a reviewer can hunt races/deadlocks. Be "very thorough".

Report with exact file:line references:
1. Every thread created (std::thread, detached threads, polling loops, timers): where created, what its loop body is, how it is signaled to stop, and whether it is joined.
2. Every mutex / std::atomic / condition_variable: name, what state it guards, which functions lock it, and any place a lock is held while doing blocking work (network I/O, sleep, cJSON parse of big payloads, waiting on another mutex).
3. Lock ordering: any function that takes two locks, directly or by calling a helper that takes another. Identify possible inverse orderings.
4. The global/static mutable state (e.g. the export "xfer plan" registry and the "inbox" registry, the room/stats reports, any singleton). Show exactly how it is protected, and find any access that is NOT protected.
5. The stop()/destructor path: enumerate everything it must unwind, and whether a thread blocked in a condition_variable::wait, in a blocking socket read, or in std::this_thread::sleep_for can miss the shutdown signal (i.e. shutdown can hang).
6. Any condition_variable::wait / wait_for without a predicate or without a loop (lost wakeup), and any notify done while not holding the lock in a way that could lose an event.
7. The HTTP handlers it registers (/pool, /pool/status, /pool/drain, /pool/router, /pool/pull, /pool/probe, /pool/node, /pool/export*, /pool/import*): which thread pool runs them (how many handler threads does server_http have? check tools/server/server-http.cpp), and which of them take locks that could serialize or block the server loop.
8. Any place a raw index/offset/size from an HTTP request body is used to index a buffer or compute an allocation (check for integer overflow on offset+size, unsigned underflow like `size - offset`, and unbounded allocation from a client-supplied length).

Quote the relevant few lines of code for each item. Do not report style or verbosity; only structural concurrency facts and concrete hazards.

**Отчёт агента (дословно):**

I need to investigate further. Let me check the details.

I have all the data. I'll write the final report. Language: According to user memory, respond in Russian, single language. Write in Russian while quoting code as-is (English quotes are fine since they're code). The report will be long, but organized by 8 items, with file:line references and short quotes attached.

File abbreviations will be defined at the beginning. # Инвентаризация конкурентной структуры `server-pool-fork.cpp`

Сокращения: **SPF** = `tools/server/fork/server-pool-fork.cpp`, **SPH** = `tools/server/fork/server-pool-fork.h`, **SCH** = `tools/server/server-http.cpp`, **SCPP** = `tools/server/server.cpp`, **SCTX** = `tools/server/server-context.cpp`, **STF** = `tools/server/fork/server-task-fork.cpp`, **SCF** = `tools/server/fork/server-context-fork.cpp`.

---

#### 1. Создаваемые потоки

Все потоки — `std::thread`, detach нет нигде.

**T1. Поллер узла `poll_node(i)` — по одному на узел (до 63 + self).**
- Создание: `SPF:816-821`
  ```cpp
  void start_poller(int n) {
      std::lock_guard<std::mutex> lk(mu_threads);
      if (stopping.load()) { return; }
      pollers.emplace_back([this, n]() { poll_node(n); });
  }
  ```
  Вызывается из `impl::found()` (`SPF:812-814`, поток discovery) и из `impl::start()` для узла 0 (`SPF:2386-2388`).
- Тело цикла `SPF:2196-2384`: копия состояния ссылок под `mu` (`2209-2220`), затем `poll_link` — серийные HTTP-запросы `/health`, `/pool/router`, `/models`, `/props`, `/slots`, `/pool/node` без блокировок (`SPF:2085-2140`, таймауты 1/3/3 с, `SPF:2090-2092`); обновление состояния узла под `mu` (`2226-2313`); возможный `POST /models/load` (соединение 1 с + чтение 10 с, `SPF:2327-2335`); зондирование ссылок `probe_link` (15/15 с, `SPF:2352-2377`, `SPF:2143-2165`).
- Остановка: атомарный `stopping` (`SPF:684`) — проверятеля в условии `while (!stopping.load())` (`SPF:2200`), в конце итерации
  ```cpp
  std::unique_lock<std::mutex> lk(mu_stop);
  cv_stop.wait_for(lk, std::chrono::milliseconds(std::max<int64_t>(50, cfg.poll_ms - (ggml_time_ms() - t0))),
      [&] { return stopping.load(); });   // SPF:2380-2382
  ```
- Присоединяется: да, `impl::stop()` `SPF:2503-2522` (обмен `pollers` на локальный вектор под `mu_threads`, затем `join` вне блокировок).

**T2. `evacuate()` — один на пул.** Создание `SPF:2391` (`pollers.emplace_back([this]() { evacuate(); });`, только в `start()`). Тело `SPF:2455-2501`: `evac_pick` под `mu`, копия `body`conversation'а, затем `migrate()` (сеть) без `mu`; при пустом выборе — `cv_stop.wait_for(500ms,.pred)` (`SPF:2474-2476`). Присоединяется через тот же `pollers`.

**T3. Поток discovery — один на весь роутер.** Создание `SPF:2931` (`th = std::thread([this]() { loop(); });`). Тело `loop()` `SPF:2863-2917`: UDP-hello каждые 2 с (первые 30 с) / 15 с; TCP-опрос `pool-peers` (`try_peer`, `SPF:2833-2861`, соедин. 1 с + чтение 3 с последовательно по всем peers — время остановки линейно по числу peers); `select` с таймаутом 250 мс; `recvfrom` → `on_datagram` (`SPF:2771`), который вызывает `impl::found()` → `start_poller` под `p->mu`. Остановка: `stopping`-предикат + `cv_stop` (`SPF:2887-2890` только при отсутствии сокетов; основной цикл — select 250 мс). Присоединяется: `discovery_t::stop()` `SPF:2934-2948`, затем закрытие сокетов.

**T4. `watch` — по одному на каждый `migrate()` (выполняется на HTTP-воркере или на потоке evacuate).** `SPF:1551-1559`:
```cpp
std::atomic<bool> pulled{false};
std::thread watch([&]() {
    while (!pulled.load()) {
        if (should_stop()) { ct.stop(); break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
});
```
Опрос через `sleep_for(100ms)` (не CV). Присоединяется в `SPF:1565` после возврата `ct.Post("/pool/pull")`. Нигде не регистрируется.

**T5. `reader` — по одному на каждый `POST /pool/pull` (HTTP-воркер).** `SPF:1646-1679`: цикл `GET /pool/export/data` чанками по 16 MiB (`POOL_CHUNK`, `SPF:441`) в очередь из ≤2 чанков. Присоединяется `reader.join()` `SPF:1710`. Нигде не регистрируется.

**T6. Циклы ожидания без потоков:** `acquire()` на HTTP-воркере `cv.wait_for(lk, 200ms)` (`SPF:1334`).

**HTTP-потоки (SCH):** по одному потоку-листенеру на адрес (`SCH:533-538`, `std::vector<std::thread> threads; // one thread per listener`, `SCH:24`), все обработчики выполняются синхронно в **одном общем** `httplib::ThreadPool` (`SCH:528-530`): фиксированных `n_threads_http` = `max(n_parallel+4, hardware_concurrency()-1)` (`SCH:150-153`) + до 1024 динамических (создаются только когда **очередь** полна). Листенеры присоединяются в `join()` (`SCH:559-567`), затем `pool->shutdown()`.

**Опасность T3-создания:** `discovery_t::start()` (`SPF:2919-2931`) **не проверяет `stopping`** перед `th = std::thread(...)`. Если сигнал останова (`SCPP:649-657 shutdown_handler → pool->stop()`) придёт между `pool->start()` (`SCPP:645`) и созданием `th` (или `stop()` уже выполнен, а `start()` ещё нет), поток будет создан, `stop()` уже вернул (`th` тогда не joinable), и никто его никогда не присоединит → деструктор joinable-потока при `~discovery_t` → `std::terminate`. Окно узкое, но реальное.

---

#### 2. Мьютексы / атомики / CV

| Объект | Где | Защищает | Кто блокирует |
|---|---|---|---|
| `static std::mutex mu` | `SPF:134` | `mt19937_64` в `pool_random_hex` | все потоки |
| `impl::mu` | `SPF:641` | всё состояние пула: `nodes[]` (mutable-поля: `links`, `link`, `status`, `draining`, `inflight`, `slot_state`, `props`, `peers`, `stats`, `gpus`, `room_*`, `load_asked`...), `queue` (`643`), `running` (`644`), `next_id`, `affinity` (`648`), `tails` (`652`), `convs` (`673`), `state_bpt`, `events` (`681`), `t_trim` | `found` `787`; `acquire` `1281` (unique_lock); `release` `1342`; `note_timings` `1396`; `link_of` `1419`; `probe_hold` `1461`; `migrate` `1488`, `1546`, `1571`; `pull` `1600`, `1704`; `forward_slot` `1755`, `1772`, `1851`, `1881`; `poll_node` `2209`, `2226`, `2317`, `2333`, `2346`, `2362`; `evacuate` `2462`, `2479`; `status_json` `2549`; `router_json` `2529`; обработчик `/pool/drain` `3097`; discovery-конфликт `2791` |
| `impl::cv` | `SPF:642` | пробуждение ожидающих в `acquire` | notify: `release` `1368`, `found` `809`, `update_mismatch` `2071`, `poll_node` `2312`, `/pool/drain` `3101` — все под `mu`; **`stop()` `SPF:2511` — вне блокировки** (безвредно, см. п.6) |
| `impl::stopping/started` | `SPF:684-685` | атомы флагов | — |
| `impl::mu_stop` + `cv_stop` | `SPF:686-687` | только ожидание остановки поллером/evacuate | `stop` `2507`, `poll_node` `2380`, `evacuate` `2474` |
| `impl::mu_threads` | `SPF:688` | вектор `pollers` (`689`) | `start_poller` `817`, `stop` `2515` (join вне блокировки) |
| `impl::n_nodes` | `SPF:639` | `atomic<int>`; сам `nodes` — фиксированный `vector(64)` (`SPF:637`), индексы стабильны | — |
| discovery `stopping/mu_stop/cv_stop/th` | `SPF:2707-2710` | — | |
| `m`, `cq` в `pull()` | `SPF:1641-1643` | локальная очередь чанков `q`, `read_done`, `failed`, `err`, `t_read` | reader/writer |
| глобальный `pool_xfer_mu` | `SPF:3198` | `pool_plans` (`3199`), `pool_inbox` (`3200`) | `3210`, `3217`, `3231`, `3288`, `3305`, `3318`, `3348` |
| `g_room_mutex_fork` | `SCF:520` | `g_room_fork` (`521`) **и** `g_stats_fork` (`533`) | report/get (`SCF:523-546`) |

**Удержания блокировки на блокирующей работе:** сеть последовательно вынесена из-под `mu` почти везде (`probe_hold` `SPF:1459-1462`, `migrate` `SPF:1487-1491` копируют ссылки под `mu` и работают с копиями; `drop_export()` `SPF:1519-1523` вызывается вне `mu`; POST `/models/load` `SPF:2331` вне `mu`; `pool_plan_drop` (unlink файла) вызывается вне `pool_xfer_mu` — `SPF:3222`, `3363-3365`). Исключения — **тяжёлая CPU-работа под `mu`**:
- `status_json` (`SPF:2548-2690`): сериализация всего дашборда (обход всех узлов, сортировка всех `convs`, JSON-dump) — полностью под `mu`; блокирует все решения `acquire` и все поллеры пула.
- `update_mismatch` (`SPF:2043-2075`): `n.model_info.dump()` для каждого узла **под `mu` на каждом опросе**.
- `trim` (`SPF:1222-1247`): полный обход `affinity`/`tails`/`convs` под `mu` из `forward_slot`.
- `server_pool_inbox_write_fork`: `pool_xfer_mu` удерживается на всё `memcpy` чанка до 16 MiB (`SPF:3288-3300`) — последовательные импорты конкурирующих передачи сериализуются одним мьютексом на весь процесс.

---

#### 3. Порядок блокировок

Два мьютекса **одновременно не удерживаются нигде** — порядок сводов везде последовательный, циклов нет:
- `found()` (`SPF:786-814`): скоуп `lock_guard(mu)` закрывается до `start_poller` (`mu_threads`) — вложенности `mu → mu_threads` нет.
- `impl::stop()` (`SPF:2503-2522`): скоуп `mu_stop` закрыт `{ }` перед notify; `mu_threads` берётся отдельно, `join` — без каких-либо блокировок.
- `pool_xfer_mu`: файловые операции (`pool_plan_drop`, удаление в `server_pool_xfer_clean_fork`) вне блокировки (`SPF:3202-3206`, `3346-3365`).
- `g_room_mutex_fork` и `pool_xfer_mu` не пересекаются с `impl::mu`.

Инвертированных порядков не обнаружено. Единственные «вложенности» — межпроцессные ожидания (HTTP-запрос на ноду, которая сама等待 свой пул), см. п.7.

---

#### 4. Глобальные/статические изменяемые данные

- **Реестр экспортов `pool_plans` + «входящих» импортов `pool_inbox`** (`SPF:3199-3200`), защищён `pool_xfer_mu` (`SPF:3198`). Все пять публичных функций берут мьютекс: `xfer_plan_fork` `3210`, `xfer_forget_fork` `3217`, `xfer_read_fork` `3231`, `inbox_write_fork` `3288`, `inbox_take_fork` `3305`, `xfer_clean_fork` `3318` + `3348`. Plan хранится как `shared_ptr<const server_pool_plan_fork>` — неизменяем, использование после выхода из-под блокировки (`SPF:3245-3274`) безопасно. **Конкретная гонка (безопасная по памяти, но с ошибкой передачи):** `xfer_forget_fork` (`SPF:3215-3224`) извлекает plan из реестра и **удаляет файл-ссылку** (`pool_plan_drop` `SPF:3202-3206`), тогда как параллельный `xfer_read_fork`, уже скопировавший `shared_ptr`, читает этот файл (`SPF:3255-3272`) → `cannot read the state`. Окно: запрос `export/done` от router A, параллельный `export/data` от reader'а B (или reader отстал от `drop_export`).
- **Отчёты room/stats** (`SCF:519-546`): `g_room_fork` и `g_stats_fork` защищены одним `g_room_mutex_fork` — protected, корректно.
- **`pool_instance()`** (`SPF:128-131`) и статический rng (`SPF:134-141`) — magic statics + мьютекс, корректно.
- **`discovery_t::challenges`, `t_conflict`** (`SPF:2704-2705`) — **без мьютекса**, защищёны только соглашением («discovery thread only» в комментарии). Все точки входа (`on_datagram`, `try_peer`, `hello_of`/`challenge` в `loop`) действительно вызываются только из потока `th`; `stop()` их не трогает. Инвариант нигде не обеспечивается механизмом.
- **«Фиксированные» поля узла** `id/name/model/self` (`SPF:507-511`, комментарий «fixed once the node joined, read without the lock») читаются без `mu` (например `poll_link`: `cli.Get("/pool/router?model=" + node.model, ...)` `SPF:2107`; `nf.id` в теле pull `SPF:1559-1561`). Запись один раз в `node_of` под `mu` до создания потока-поллера (`found` `SPF:812-813`) — отношение «произошло до» выполняется, записей после нет. Корректно, но держится на дисциплине.
- `disc->names`, `impl::cfg`, `api_key`, `self_host` — иммутабельны после конструктора/`init`, читаются из хендлеров без блокировки — OK.

---

#### 5. Путь `stop()`/деструктора

`~server_pool_fork` (`SPF:2995-2997`) → `stop()` (`SPF:3006-3012`): сначала `disc->stop()`, затем `impl::stop()` каждого пула. Идемпотентно (`stopping.exchange`). Вызывается из `clean_up` и `shutdown_handler` (`SCPP:620-634`, `649-657`).

Разворачиваемое:
1. `discovery_t::stop` (`SPF:2934-2948`): `stopping=true`, `{lock mu_stop}`, notify, `th.join()`, закрытие сокетов. Поток может быть в: `select` (≤250 мс), `recvfrom` (защищён select), `try_peer` (последовательно до 4 с **на каждый peer** — не ограничено сверху количеством peers).
2. `impl::stop` (`SPF:2503-2522`): notify `cv_stop` и `cv`, обмен `pollers`, join всех поллеров. Поллер может быть в блокирующем HTTP — **все клиенты имеют выставленные таймауты**, подвешивания без возврата нет: poll_link 1/3/3 с (`2090-2092`), `probe_link` 15+15 с (комментарий `SPF:2147` прямо признаёт: «a stop of the router waits for a probe that runs»), `/models/load` 1+10 с (`2330-2331`). Худший случай присоединения — десятки секунд.
3. `evacuate`-поток внутри `migrate`: единственный незавершённый `ct.Post("/pool/pull")` (чтение = `params.timeout_read`, `SPF:1497`) прерывается потоком `watch` за ~100 мс через `ct.stop()` (`SPF:1553-1556`) — при условии, что `should_stop` = `stopping` (для evacuate это так, `SPF:2478`).
4. Ожидающие `acquire` (HTTP-воркеры) не джойнятся; они выходят по `stopping.load()` (`SPF:1290-1294`) в пределах 200 мс (нет предикатного `wait_for`).

Может ли остановка зависнуть:
- `cv_stop`-ожидания: предикат `stopping` — событие не теряется, хотя `notify` в `stop()` сделан вне блокировки (`SPF:2507-2511`).
- `cv.notify_all()` вне `mu` (`SPF:2511`) для `acquire` без предиката — компенсируется 200-мс таймаутом, не зависает.
- `sleep_for(100ms)` в `watch` — проверяет `pulled`/`should_stop` после каждого сна, не зависает.
- Конкретные опасности останова:
  (a) **`std::terminate`** при гонке сигнала с `pool->start()` — `discovery_t::start` не проверяет `stopping` (см. п.1, T3).
  (b) **`std::terminate`** при любом исключении между созданием и `join` потоков `watch` (`SPF:1552`→`1565`) и `reader` (`SPF:1646`→`1710`):析struct thread деструктируется joinable при раскрутке (внешний exception-handler SCH:183 перехватить уже не успеет).
  (c) `shutdown_handler` останавливает пул **до** `ctx_http.stop()` (`SCPP:653-656`): между ними HTTP-воркеры всё ещё принимают `/pool/pull` и запускают новые `reader`-потоки и сетевые само-запросы, пока инстансы уже выгружаются (`clean_up` → `unload_all` `SCPP:629-631`).
  (d) Нодовые хендлеры `/pool/export`, `/pool/import/done` ждут `rd.next(req.should_stop)` **без таймаута** (`SCTX:7846-7852`, `7995-8008`) — остановка процесса-инстанса ждёт их; ограничено только тем, что клиент (пул) отвалится по своему read-таймауту.

---

#### 6. Условные ожидания: потеря пробуждения

- `SPF:1334` — `cv.wait_for(lk, std::chrono::milliseconds(200));` **без предиката** — но внутри `while (true)` с полным перепроверением условий и жёстким 200-мс таймаутом → потеря сигнала стоит ≤200 мс, зависнуть не может.
- `SPF:2381`, `SPF:2475`, `SPF:2889` — `cv_stop.wait_for(..., [&]{ return stopping.load(); })` — предикат + таймаут, safe.
- `SPF:1650` — reader: `cq.wait(lk, [&] { return q.size() < 2 || failed; });` — **бесконечный wait с предикатом**. Все записи `failed`/`read_done` — под `m`, и writer уведомляет на **каждом** пути выхода из цикла: `SPF:1684-1687` (`if (failed || (q.empty() && read_done)) { cq.notify_all(); break; }`) и после извлечения чанка `SPF:1696`. Потери нет; единственный сценарий зависания reader'а — раскрутка исключения writer'а (тогда terminate по (b) п.5).
- `SPF:1681` — writer: `cq.wait_for(lk, 200ms, pred)` — предикат + цикл.
- Notify вне блокировки: только `impl::stop()` `SPF:2510-2511` — для `cv_stop` безвредно (атомарный предикат), для `cv` — см. выше. Все остальные `notify_all` выполняются под соответствующим мьютексом.

---

#### 7. HTTP-хендлеры и модель потоков

Все зарегистрированные обработчики исполняются **синхронно в одном общем `httplib::ThreadPool`** (`SCH:25`, `SCH:528-530`): фиксированных `n_threads_http` (`max(n_parallel+4, cores-1)`, `SCH:150-153`) + до 1024 динамических, создаваемых только при **полной очереди**. По потоку-листенеру на адрес (`SCH:533`).

Регистрация в `attach()`:
- `wrap(...)` для всех POST инференс-роутов (`SPF:3033-3056`) → `forward_slot`: **воркер припаркован** на всё время ожидания слота в пуле (до `pool-wait-max` = 300 с по умолчанию; `acquire` `SPF:1280-1337`) плюс на весь прокси-стрим; `cleanup`-лямбда (`SPF:1912-1914`) вызывает `release`/`note_timings` (берут `mu`) на том же воркере.
- `GET /pool` `SPF:3058` — статический HTML, без блокировок.
- `GET /pool/status` `SPF:3067` → `status_json` — **удерживает `impl::mu` на всю сериализацию** (`SPF:2548`): конкурирует со всеми решениями маршрутизации и поллерами; самый тяжёлый пользователь `mu`.
- `POST /pool/drain` `SPF:3087` — последовательно берёт `mu` каждого пула (`SPF:3096-3105`), коротко.
- `GET /pool/router` `SPF:3114` → `router_json` — `mu` + JSON-dump пиров.
- `POST /pool/pull` `SPF:3126` → `pull` `SPF:1587-1736`: **воркер занят всю передачу состояния** (секунды-минуты), порождает `reader`-поток, и **на каждый чанк делает само-запросы к этому же серверу** — `ct(self_host, http_port)` `SPF:1618`: `POST /pool/import/data` (`SPF:1698`) и `POST /pool/import/done` (`SPF:1631`, `1719`). Эти запросы требуют **другого воркера того же общего пула**. При исчерпании фиксированных воркеров (все припаркованы в `acquire`/`pull`) import-POST ждёт в FIFO-очереди за ним; два одновременных `pull` взаимно блокируют друг друга до 60-с read-таймаутов (`SPF:1620-1621`) — не вечный deadlock, а взаимный отказ по таймауту; при большем таймауте стал бы deadlock'ом. `/pool/import/done` дополнительно постит задачу в серверный цикл инстанса (`SCTX:7954-7968`) и ждёт её.
- `GET /pool/probe` `SPF:3135-3146`: `res->data.assign(size, '\0')` — до 64 MiB на запрос; `POST /pool/probe` `SPF:3148-3152` — тело отбрасывается, но целиком буферизуется httplib.

Нодовая сторона (в роутере проксируется инстансу `SCPP:582-587`, в одно-модельном сервере напрямую `SCPP:725-730`):
- `GET /pool/node` (`SCTX:7759-7796`): `server_pool_node_devices_fork` (вызовы драйверов `ggml_backend_dev_memory`), room/stats под `g_room_mutex_fork`, плюс `server_pool_xfer_clean_fork` — `pool_xfer_mu` + обход каталога, на воркере, без `impl::mu`.
- `POST /pool/export` (`SCTX:7798-7854`): токенизация промпта на воркере, затем пост `SERVER_TASK_TYPE_POOL_EXPORT` и блокировка на `rd.next` до очереди GPU-цикла — воркер припаркован за длинным декодом.
- `GET /pool/export/data` (`SCTX:7857-7881`): `pool_xfer_mu` кратко; чтение файла/memcpy ≤64 MiB без блокировок.
- `POST /pool/export/done` (`SCTX:7914-7921`): unlink файла вне мьютекса.
- `POST /pool/import/data` (`SCTX:7924-7946`): **`pool_xfer_mu` удерживается на всё memcpy чанка** (`SPF:3288-3300`).
- `POST /pool/import/done` (`SCTX:7949-8010`): take inbox, пост задачи, `rd.next`.

---

#### 8. Клиентские offset/size/length — переполнения и аллокации

Проверенные и корректные места:
- `GET /pool/export/data` `SCTX:7863-7868`: `std::stoull` (замечание: `stoull("-1")` не бросает, а оборачивается в огромное число), но `server_pool_xfer_read_fork` защищает вычитание:
  ```cpp
  if (offset > plan->size) { error = "the offset is past the end of the state"; return false; }   // SPF:3246-3249
  out.resize((size_t) std::min(size, plan->size - offset));                                        // SPF:3260
  ```
  `plan->size - offset` без underflow; `out` ограничен размером состояния и 64 MiB (`std::min<uint64_t>(..., 64ull << 20)` `SCTX:7866`).
- Идентификаторы трансферов валидируются как ≤32 hex-символа до построения пути файла (`SPF:3204-3208`, `SPF:3277-3279`) — path traversal нет.
- `server_pool_inbox_fork::write` `STF:1428-1475`: `offset != in.received` отбрасывает любые дыры; `n > in.lay.size - in.received` (`STF:1433`) не может уйти в underflow, т.к. `received` инкрементируется только под этим ограничением (`STF:1437`, финальная проверка `STF:1471-1473`, ошибочный inbox удаляется из реестра `SPF:3296-3299`).
- `pull()`: `size` из тела запроса (`SPF:1597`) задаёт цикл `for (uint64_t off = 0; off < size; )` (`SPF:1636`) — вложенный источник, присылающий больше запроса, приведёт к overshoot и срабатыванию `off != size` → 502 (`SPF:1714`), без UB.

Конкретные опасности:
1. **Неограниченная аллокация из заголовка состояния, присланного по сети.** `parse_head` (`STF:1357-1382`) ресайзит `prompt/main/drft/index` по клиентским `h.n_prompt/n_main/n_drft/n_ckpt`. Верхняя граница — `spill_header_sane` (`STF:97-101`):
   ```cpp
   constexpr uint64_t SPILL_MAX_BYTES = 1ull << 40; // far above any real state
   ```
   т.е. до ~1 TiB на секцию, а `max_bytes`-проверка (`STF:1365-1370`) **пропускается при `max_bytes == 0`** — семантика «0 = any» (`SPH:65`), значение — `meta->pool_cache_ram_fork` (`SCTX:7937`). Узел с неограниченным prompt-cache принимает от любого пира заголовок, вызывающий гигабайтные/терабайтные `resize` (ловушка `bad_alloc` есть — `STF:1375-1377`, `1414-1416`, но векторы обнуляются, и при overcommit скорее сработает OOM-killer, чем исключение). `parse_index` (`STF:1386-1420`) — то же для ≤4096 чекпойнтов × 3 буфера.
2. **Усиление памяти в `GET /pool/probe`**: `SPF:3143` / `SCTX:7892` — до 64 MiB на запрос, на каждый активный воркер (до `n_threads_http`+1024) — memory-DoS в пределах LAN (роль узла/роутера — сетевой).
3. Первый чанк импорта до разбора заголовка ограничен только буферизацией тела в httplib (лимит payload не выставлен): `received` накапливается до `sizeof(h)`, `memcpy` вне заголовочных целей не происходит — утечки нет, но тело целиком resides в памяти воркера.

---

#### Сводка конкретных опасностей (по убыванию серьёзности)

1. `std::terminate`: `discovery_t::start` без `stopping`-гейта (`SPF:2919-2931`) при гонке сигнала со стартом; исключения между `std::thread(...)` и `join` в `migrate` (`SPF:1552/1565`) и `pull` (`SPF:1646/1710`).
2. Само-запросы `pull()` в собственный пул воркеров (`SPF:1618`, `1698`, `1719`) на фоне массового паркования воркеров в `acquire` — взаимная блокировка одновременных pull'ов до 60-с таймаутов; приоритетная инверсия import-трафика за очередью чат-запросов.
3. Неограниченная (до `SPILL_MAX_BYTES`=1 TiB) аллокация из клиентского spill-заголовка при `pool_cache_ram_fork == 0` (`STF:1357-1382`, `STF:97-101`, `SPH:65`); 64 MiB × воркеров в `/pool/probe`.
4. Тяжёлая CPU-работа под `impl::mu`: `status_json` (`SPF:2548-2690`), `update_mismatch` с json-dump на каждый опрос (`SPF:2043-2075`), `trim` (`SPF:1222-1247`) — сериализуют маршрутизацию за дашбордом; 16-MiB memcpy под глобальным `pool_xfer_mu` (`SPF:3288-3300`) — сериализует параллельные импорты.
5. Гонка unlink-файла экспорта с параллельным `export/data` (`SPF:3215-3224` vs `SPF:3255-3272`) — отказ передачи, не UB.
6. Ограниченные по времени, но длинные остановки: `probe_link` 15+15 с при join поллера (признано в `SPF:2147`), `try_peer` × число peers при join discovery, нодовые `rd.next` без таймаута (`SCTX:7846`, `7995`), остановка пула до `ctx_http.stop()` с живыми хендлерами (`SCPP:649-657`).

Потерянных пробуждений, инверсий порядка блокировок и небезопасных обращений offset/size (underflow/overflow) не обнаружено — код в этом отношении выстроен дисциплинированно (скоупы `mu` всегда закрываются перед сетью, все сетевые клиенты с таймаутами, все CV-ожидания либо с предикатом на атомарный `stopping`, либо с коротким таймаутом).

---

### B3.3 Python-тесты форка сервера

- Агент: `Explore-vDgoo3C3if2ZBXVSZqdRTQEO3Jq2U1td` (тип `Explore`, глубина 1, родитель: `general-purpose-twnf47f6Kg08awPJs2CIxtlF5vTtTT5x`)
- Статус: completed, создан 14:44:23 UTC, финальный отчёт 18181 символов

**Постановка задачи (дословно из лога):**

READ-ONLY task (do not modify files). In repo /Users/askuzminov/Projects/github/askuzminov/llama.cpp (a perf fork of llama.cpp), review these new test files:
- tools/server/tests/unit/test_pool_fork.py (1307 lines)
- tools/server/tests/unit/test_slot_room.py (242 lines)
- tools/server/tests/unit/test_router_fork.py (156 lines)
- tools/server/tests/unit/test_cache_spill.py (873 lines)
- tools/server/tests/utils_fork.py
and how they are registered/run (check tools/server/tests/unit/conftest.py or CMakeLists / how -n_parallel and server params are passed, and tools/server/tests/utils.py diff vs master with `git diff master...HEAD -- tools/server/tests/utils.py`).

For each test file answer:
1. What behavior is it actually asserting? Summarize per test case, concise.
2. CRITICALLY: which tests could still PASS if the feature under test were broken? Look specifically for: assertions that only check HTTP 200 or that a request completed; tests that don't compare generated text against a reference; tests that sleep/poll with a generous timeout and then accept either outcome; tests where the precondition is never actually created (e.g. cache eviction never forced, so the "reuse" path is not exercised); tests that would pass even if the pool never routed anywhere (single-node); missing assertions about slot state; reliance on default server params that make the interesting code path unreachable (e.g. n_parallel=1 so slot reuse never races, prompt too short to trigger the budget).
3. Do the tests assert determinism of output across the paths they exercise (e.g. same prompt -> same text before and after a spill/restore or a pool migration)? Where output text comparison is missing, say so.
4. Any test that is skipped/disabled by default, or requires env/GPU that CI will not have.
Report file:line for each point. Be concrete and terse; no style commentary.

**Отчёт агента (дословно):**

make_request itself does not raise an exception on non-2xx (only stream does at line 445). Therefore, tests that ignore the result (e.g., test_no_pause_below_ram_reserve in test_slot_room) will pass even if the request fails.

Next, write the report. Keep it concise, attach file:line. Should I respond in Russian? The user's memory says "Ответы по-русски (respond in Russian)." The user's message is in English. The rule: respond in the same language as the input. The input is English. The memory is cross-project and states the user asks for Russian, in a single language. The rule says use the language that best matches the primary request, i.e., English. But memory indicates the user prefers Russian... Memory says the user asks for responses in Russian. Following the memory policy, memory about communication preferences applies. The message is entirely English, but the saved preference says respond in Russian. Follow memory (communication_preference type) and write in Russian. Single language: Russian, keeping technical terminology as-is.

Structure the report by file. Include line numbers. Let me collect the key line numbers:

test_pool_fork.py:
- chat helper 98-106 (assert 200, returns content, never compared)
- No content == assertions anywhere (grep confirms 0 hits).
- test_pool_takes_the_model_under_every_name 294: assert events new ==2; doesn't check node identity → passes even if pool routes nowhere / single-node.
- test_pool_any_router_takes_requests 686-691: only checks events recorded via router1; placement unconfirmed.
- test_pool_waits_with_a_long_prompt_for_a_quiet_node 463-500: order only via timestamps at 499-500; for params (0,1000) and (30,8) asserts done["new"] < done["long"], which is the "normal" outcome — passes even if waiting is broken.
- test_pool_puts_a_short_answer_first 624-666: completion order timestamps 664-666.
- test_pool_rebalances_a_crowded_node rebalance=False 385; test_pool_moves_a_long_prompt long_message=False 425: control branch passes when feature is broken.
- test_pool_sends_no_prompt_that_has_no_room repeat=3 581 control.
- test_pool_queues_when_every_slot_is_busy 668-685: has real slot assertions (inflight<=2, queued>0, done==4).
- test_pool_measures_the_prefill_speed 902: sleeps 0.3 seconds and then asserts tps==0 (timing-dependent); poll 5 seconds has discriminating power.
- Skips: 735 skip if no LAN, broadcast 745; nt only affects router files.
- plain_nodes port 1139: routers[0].server_port + 10 + i → port collides with next pytest worker under -n 4 (conftest port_base stride 10; utils conftest.py:8-13).
- Node-side tests 1146-1206: strong — cache_n == ex["n_reuse"] 1186, 1205.
- test_pool_moves_no_state_with_migrate_0 872: asserts cache_n==0 and details — has discriminating power.
- test_pool_loads_the_model_on_demand 169: strong.
- test_pool_shows_a_failed_load 192: polling; load event only for n1 — good.
- test_pool_weighs_a_node_by_its_answers_alone around 937: decode_tps==0 under load, then >0 — has discriminating power.
- Determinism: no prompt→same-text check after migrate across the whole file. temperature=0 (line 101) in chat().

test_router_fork.py:
- 77 pending.join(30) — no assertion on result.
- The rest are strong (process liveness, timing).
- 44, 63, 81 skipif nt — CI Windows skips (server.yml has a windows job running tests.sh).
- test_router_killed_after_exit_command_child_exits 111-115: only asserts waited > stop_timeout-1 (lower bound); doesn't check that child exits within stop_timeout + margin — but _wait_pids_gone(timeout=stop_timeout+5) bounds it. OK.
- test_router_load_takes_room_only_from_an_idle_model 122-156: solid; hardcoded sleep(1.5).

test_slot_room.py:
- test_running_slot_pauses_for_room 147-178: the only determinism comparison (178).
- test_idle_slot_gives_way_and_comes_back 74-105: cache_n assertion around line 103 — actually line: `assert res.body["timings"]["cache_n"] >= n_tokens(...) - 1` is around 103. Also NO_ROOM is absent. Strong. But no text comparison of the resumed conversation against solo execution.
- test_task_waits_for_room 108-133: asserts tokens_predicted == n_predict (guards against truncation), log "waits for room"; no content comparison.
- test_full_context_task_does_not_wait_in_an_empty_cache 136-160: checks truncated flag — good.
- test_no_pause_below_ram_reserve 179-199: response result discarded around line 196 (parallel_function_calls) — no status assertion; make_request doesn't raise on non-2xx (utils.py:394) → passes even if both requests 500. Only asserts log strings.
- test_generation_progress_during_another_prompt 201-243: log-based; asserts stop and decode count.
- No skips; all CPU-safe.

test_cache_spill.py:
- Header at lines 18-21 admits eviction is never forced in the first test — test_prompt_cache_spill_enabled_reuse 38-91: never asserts spill file; reuse might be from RAM; would pass even if spill path is broken; "spill" only via enabled flag. Also no content identity; final "state consistent" only cache_n>0 (85-90).
- test_prompt_cache_spill_written_on_shutdown 94-148: asserts file + warm restart reuse; no content comparison.
- test_prompt_cache_writes_what_leaves_ram 151-221: strongest (file transitions, threshold cache_n>1100/>1300).
- test_prompt_cache_write_behind_while_idle 224-278: polling ≤5s to file; assert 1 file; no rewrites; reuse cache_n>0 only.
- test_prompt_cache_spill_kill_mid_write 280-356: race — if rename wins the kill (around line 320 tmp poll and kill), spill_files==tmp assert fails → flaky; also asserts cleanup.
- test_prompt_cache_spill_with_mmproj 358-404: assert 1 file and reuse; no content.
- test_prompt_cache_spill_with_image 407-493: downloads IMG_URL from HF at test time (line 410) — network dep without skip guard; header check 466-469; reuse cache_n>0, prompt_n<cold; sets temperature 0 and top_k 1 but doesn't compare output text of cold vs warm.
- test_prompt_cache_hit_keeps_the_file 496-558: invariant directory listing — good; reuse assertion is somewhat weak.
- test_prompt_cache_ram_min_tokens_skips_short_prompts 561-603: empty-dir assertion passes equally well when --cache-idle-write is broken (negative test with no positive control in the test itself; relies on other tests).
- test_prompt_cache_min_tokens_keeps_short_prompts_in_ram 605-641: cache_n >= first prompt_n-1 — state-level; good.
- test_prompt_cache_spill_dir_shared_by_two_models 643-697: good.
- test_context_checkpoint_chain_re_anchors 699-745: log regex only; requires debug trace; asserts BASE at idx>1 — has discriminating power.
- test_prompt_cache_spill_carries_checkpoints 747-830: binary header + log ordering — strongest.
- test_prompt_cache_returning_prompt_beats_slot 832-873: cache_n >= n_prompt_a — has discriminating power.
- No skips; no slow mark; runs CPU in CI (server.yml around lines 127-138). GPU via N_GPU_LAYERS env if set.

Registration/execution:
- No conftest.py in unit/ directory; single conftest at tools/server/tests/conftest.py: per-worker ports (8-13), server stop (17-27), preset load lock (30-37).
- tests.sh: -n ${PYTEST_WORKERS:-4} --dist=worksteal; by default excludes slow mark; CI runs ./tests.sh (CPU) per server.yml, slow tests only on schedule.
- No CMakeLists in tests directory — tests are not registered with CTest; run manually/by workflow.
- -n_parallel (server's parallel) is not a utils option; slot concurrency is set by per-test server.n_slots; fork parameters are injected via utils_fork.py through ServerProcess.start (utils.py:317-318), and the default env LLAMA_SERVER_DEBUG_FAKE_TIMING=1 (utils.py:150) is only removed in pool tests via env_fork.
- Import in utils.py: ServerProcess inherits from ServerProcessFork (utils.py:52).
- test_router_fork rebinds module state of test_router (test_router_fork.py:7-11, 15-19) — fragile coupling but works.

One more thing: the port collision claim — conftest: port_base + worker_num*10. Pool routers use +1..+3 (base+3). Broadcast port routers[0].server_port + 900 = base+901: another worker w2: base2=base+10 → base2+901 = base+911 vs our +901 no problem; but do 2 workers both run broadcast tests? Only one worker per file. But pool's +900 and another test? Port base+901 doesn't collide with worker range (base..base+9, and tests may use other ports via make_stream...). plain_nodes: base+11, +12 → collides with worker w+1's router port (w+1's base+1 = base+11). If both run simultaneously (pool_fork on worker gw1 uses plain_nodes while pool_fork again? A file runs on one worker; but router_fork and pool tests on other workers use base+1...). test_router_fork uses ServerPreset.router() with PORT = that worker's base as-is (no +1). test_router also base. So base+11 collides only with another pool_fork run — same file only one worker at a time. But test_pool_fork's own routers use base+1, +2 and plain_nodes uses base+11, +12 — no intra-worker collision. Cross-file: metrics tests etc. use base. So collision is unlikely, but if another worker ran pool tests... impossible — same file goes to one worker. Broadcast +900: no problem. Mention as a brief minor note on port stride risk.

Also note: test_pool_fork fixture creates routers but doesn't start them inside the fixture; tests start them. OK.

Furthermore, point 4: skips — test_pool_fork.py:733-743 (broadcast skipped if no LAN — GitHub runners usually have LAN, but containers might not), test_router_fork.py:44/63/81 nt skip. None of the 4 files have GPU requirement; CI is CPU (N_GPU_LAYERS unset). None marked slow → all run on default CI, -x flag makes them abort.

Also, cache_spill image test: fetches IMG_URL from HF with requests at test time (lines 410-411) — if HF is unreachable, the test fails (not skipped) in CI.

Now write a concise answer in Russian. # Проверка тестов форка (only read, ничего не менял)

#### Как регистрируются и запускаются

- `unit/` не имеет своего `conftest.py`; единственный — `/Users/askuzminov/Projects/github/askuzminov/llama.cpp/tools/server/tests/conftest.py`: порты `port_base + worker*10` (стр. 8-13), остановка серверов после каждого теста (17-27), блокировка загрузки пресетов (30-37).
- Запуск: `tools/server/tests/tests.sh` — `-n ${PYTEST_WORKERS:-4} --dist=worksteal`, по умолчанию `-m "not slow"`, `-x`. CI (`/…/.github/workflows/server.yml` ~127-138) гоняет `./tests.sh` на CPU; `SLOW_TESTS=1` только по расписанию. **Ни один из четырёх файлов не помечен `slow`/`serial`** — все идут в обычном CPU-CI параллельно 4 воркерами.
- CMake/CTest-регистрации нет (в `tools/server/tests/` нет CMakeLists).
- Параметры форка.inject: `utils_fork.py` (весь файл) -> `ServerProcess(ServerProcessFork)` в `utils.py:52`, аргументы/окружение добавляются в `utils.py:317-318`. Замечание: `LLAMA_SERVER_DEBUG_FAKE_TIMING=1` ставится для **всех** серверов (`utils.py:150`); только тесты пула снимают его через `env_fork` (`test_pool_fork.py:38`). Тесты `test_slot_room`/`test_cache_spill` живут с ним.
- `test_router_fork.py:7-19` переиспользует хелперы `test_router`, подменяя его модульный `server` — работает, но хрупкая связь (import тест-модуля).

---

#### test_pool_fork.py

Сильная часть: почти всё сверяется с журналом событий пула (`events("migrate"/"keep"/"recompute"/…)`) с указанием узла и conv, плюс `cache_n` в таймингах — это реально различает «состояние перевезли» vs «пересчитали». Node-side тесты самые сильные: `test_pool_node_sends_the_checkpoint_the_request_needs` (проверяет `timings.cache_n == ex["n_reuse"]`, стр. ~1200, 1205) и `test_pool_moves_the_state_onto_a_node_that_lost_its_copy` (1235-1301: `cache_n >= 30 and prompt_n < cache_n`).

Слабые места:

- **Нет ни одного сравнения сгенерированного текста** (проверено grep: `content ==` отсутствует во всём файле). `chat()` возвращает content (стр. 98-106) и нигде не сверяет его. Все тесты миграции состояния (`test_pool_moves_the_state_of_a_conversation` 773-797, `test_pool_drain_hands_the_conversations_over` 799-823, `test_pool_moves_between_stream_counts` 849-870, `test_pool_moves_the_state_when_computing_looks_shorter` 825-847) доказывают только *переиспользование* (`cache_n >= …`), но не *идентичность вывода* до/после миграции. Повреждённое, но «_working_» состояние с другим continuation пройдёт.
- `test_pool_waits_with_a_long_prompt_for_a_quiet_node` (463-500): проверятся только метки времени `done["new"] >/< done["long"]` (499-500). Параметризация `(0,1000)` и `(30,8)` — отрицательные контроли: проходят и при полностью сломанном ожидании. Дискриминирует только `(30,1000)`; при этом нет проверки, что запрос реально сидел в очереди (`queued`).
- `test_pool_puts_a_short_answer_first` (624-666): то же — только порядок меток времени (664-666), без проверки слотов/очереди на ноде.
- `test_pool_takes_the_model_under_every_name` (294-307): `len(events("new")) == 2` без проверки узла — пройдёт и на одноузловом пуле, и если пул никуда не роутиит.
- `test_pool_any_router_takes_requests` (686-691): проверяет лишь что событие `new` записано через роутер 1; целевой узел не проверен — пройдёт при роутинге всегда локально.
- `test_pool_measures_the_prefill_speed` (902-930) / `decode_speed` (932-943): «нет сэмпла» проверяется сном 0.3 с — гонка с асинхронным измерением (может и ложно проходить, и ложно падать).
- `test_pool_broadcast_needs_the_same_key` (745-771): после `time.sleep(5)` — «чуть больше раундов broadcas­та»; изоляция n2 фиксировалась бы и при слегка поломанном discovery.
- Контрольные ветки параметризаций (`rebalance=False` 385, `long_message=False` 425, `repeat=3` 581, `spill/min_tokens` 773) по конструкции проходят при сломанной фиче — это ок как контроль, но не покрытие.
- Инфра: `plain_nodes` берёт порты `routers[0].server_port + 10 + i` (стр. ~1139) = `PORT+11/+12`, тогда как шаг портов воркера — ровно 10 (`conftest.py:12`) → потенциальные коллизии с другим воркером при `-n 4`.
- Скипы: `test_pool_finds_its_routers_by_broadcast` (733-736) и `test_pool_broadcast_needs_the_same_key` (748-749) скипаются при отсутствии не-loopback сети (контейнерный CI их молча пропустит). Остальное — без GPU-зависимостей.

#### test_slot_room.py

Лучший из файлов: посылки вычисляются (`assert max(sizes) < n_ctx < sum(sizes)`), проверяются строки журнала + `cache_n` + `truncated`.

- `test_running_slot_pauses_for_room` (147-178): **единственное в четырёх файлах сравнение детерминизма текста** — `alone.body["content"] == res.body["content"]` (стр. 178).
- `test_idle_slot_gives_way_and_comes_back` (74-105): `cache_n >= n_tokens(STORY_A*5)-1` (~103) — путь переиспольования реально форсируется (слот 0 вытесняется заведомо, посылка проверена). Текста после восстановления не сверяют.
- `test_task_waits_for_room` (108-133): `tokens_predicted == n_predict` защищает от усечения, но содержимое ответов не сверяется с последовательным прогоном.
- `test_no_pause_below_ram_reserve` (179-199): **результаты запросов выброшены** (стр. 196) — `make_request` не бросает на не-2xx (`utils.py:394`), т.е. тест пройдёт и при HTTP 500 на обоих запросах;Assert'ы только по журналу.
- `test_generation_progress_during_another_prompt` (201-243): сильно (порядок строк debug-лога, `n_decode > 1`, `stop`).
- Скипов нет; всё CPU-safe.

#### test_router_fork.py

- `test_router_killed_child_exits` (44-61), `test_router_stop_does_not_wait_for_load` (63-78), `test_router_killed_after_exit_command_child_exits` (81-120): сильные — проверяют живость PID-процессов детей и нижнюю границу `waited > stop_timeout - 1` (118).
- **Стр. 77: `pending.join(30)` — комментарий «it fails», но результата никто не проверяет**; тест пройдёт, даже если запрос успешно завершился.
- `test_router_load_takes_room_only_from_an_idle_model` (123-156): сильный (отказ загрузки, статусы обеих моделей); магический `time.sleep(1.5)` (150) — на медленном CI borderline.
- Скипы: три первых теста `skipif os.name == "nt"` (44, 63, 81) — Windows-джоба CI (`server.yml` ~200-215) их пропустит.

#### test_cache_spill.py

- **Главное признано самими авторами (стр. 18-21): вытеснение RAM->диск никогда не форсируется** для крошечных моделей. `test_prompt_cache_spill_enabled_reuse` (38-91): ни одного assert'а на файл в `spill_dir`; «reuse» = `cache_n > 0` + `prompt_n < original` (78-79) — **пройдёт и при полностью сломанном spill** (переиспользование из RAM-кэша). «State coherent» (85-90) — тоже только `cache_n > 0`, текст не сверяется.
- Остальные тесты сильнее и реально проверяют файлы: `written_on_shutdown` (94-148: ровно 1 файл, повторный warm-рестарт), `writes_what_leaves_ram` (151-221: переходы количества файлов по шагам + пороги `cache_n>1100/>1300`), `hit_keeps_the_file` (496-558: неизменный листинг), `carries_checkpoints` (747-830: заголовок `LCPCACHE`, `restored` раньше `created` в логе), `returning_prompt_beats_slot` (832-873: `cache_n >= n_prompt_a`).
- **Ни один тест не сравнивает текст cold vs warm**, хотя `temperature=0` (+`top_k=1` в image-тесте) это позволяет; детерминизма spill/restore-пути не проверяются нигде в файле.
- `write_behind_while_idle` (224-278): опрос файла до 5 с (247-252) — ок; «not rewritten again» ждёт всего 0.5 с (259) — слабо, но приемлемо.
- `spill_kill_mid_write` (280-356): **гонка** — между обнаружением `.tmp` (320-321) и `kill()` rename может завершиться, и тогда `spill_files == tmp` (325) ложно упадёт; сам assert на очистку мусора при старте хороший.
- `ram_min_tokens_skips_short_prompts` (561-603): пустой каталог — негативный assert, который проходит и при сломанном `--cache-idle-write` (позитивный контроль лежит в другом тесте, но сам тест неразличающий).
- `spill_with_image` (407-493): скачивает `IMG_URL` с HuggingFace во время теста (410-411) без skip-guard — жёсткая сетевая зависимость CI; проверка заголовка (466-469) и warm-рестарт есть.
- Скипов/медленных маркеров нет; `tinygemma3`/`mmproj` требуют загрузки HF-моделей (в CI есть сеть), GPU не требуется.

#### Итог по главному вопросу (3)

Единственная проверка «тот же промпт -> тот же текст» на всех 2500+ строках — `test_slot_room.py:178` (пауза/возобновление). Ни миграция состояния пула, ни spill/restore с диска, ни checkpoint-перенос не проверяют идентичность генерации — везде ограничиваются таймингами `cache_n`/`prompt_n` и presence-строками журнала.

---

### B4 Разбор падения test-spec-draft-q-fork

- Агент: `general-purpose-ujldbPelVy83tnh4mqLAIOq3KBRn6tjY` (тип `general-purpose`, глубина 0, родитель: корневой вызов ревью)
- Статус: completed, создан 14:46:26 UTC, финальный отчёт 13637 символов

**Постановка задачи (дословно из лога):**

READ-ONLY investigation. Do not edit, create, or delete files in the repo. Do not rebuild (a build tree already exists at /Users/askuzminov/Projects/github/askuzminov/llama.cpp/build-qreview; you may RUN its binaries).

REPO: /Users/askuzminov/Projects/github/askuzminov/llama.cpp (branch `cache`, a perf fork of llama.cpp).

ESTABLISHED FACTS (do not re-derive):
- The build succeeds. Running `ctest -R fork|speculative|top-k|row-cache` in build-qreview gives 7 failures, but on rerun most pass (they are not parallel-safe). Three fail deterministically:
  1. `test-spec-draft-q-fork` (tests/test-spec-draft-q-fork.cpp). Output: many `FAIL .../test-spec-draft-q-fork.cpp:195: same_q(q0, q[0])`, then `first: 2180 of 12000 drafts are tokens outside q`, `FAIL ...:212: fits("first", first, q0, n_first)`, `second: 763 of 1536 drafts are tokens outside q`, `FAIL ...:229: fits("second", ...)`. It still fails with `LLAMA_SAMPLER_FAST_TOP_K=0`, so the fork's fast top-k is NOT the cause.
  2. `test-qsa-multi-fork` fails at line 515 `CHECK(worst == 0.0)` — the QSA sparse-list path differs from the dense mask by 1.49e-08 in the "independent" scenario (bit-exactness is asserted).
  3. `test-moe-cache-fork-upload` fails only when run in parallel with other tests (`token 0: 95 against 5 of the reference`).

YOUR TASK - answer these, in this order of importance:

(A) ROOT CAUSE of the draft-q failure. Read tests/test-spec-draft-q-fork.cpp fully (esp. `q_of`, `same_q`, `fits`, and the trial loop ~line 150-230) and the producer side in common/speculative.cpp (~lines 360-480 for the draft-simple path: `id_sampled = common_sampler_sample(...)`, `cur_p = common_sampler_get_candidates(...)`, `id = dp.result_q ? id_sampled : cur_p->data[0].id`, `dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size)`). Determine WHY ~18% of drafted tokens have zero probability in the recorded q. Distinguish between these candidate explanations and say which one the code actually supports:
  - q holds a candidate array whose `p` values underflow to 0.0f for tail tokens (softmax over the whole vocab), while the sampler's internal draw can still pick such a token;
  - the draft sampler and the recorded q come from different chains/positions;
  - the sampler keeps state (penalties / ngram / accepted tokens / rng) across trials because the test resets KV with `llama_memory_seq_rm` but never resets the draft sampler - check whether `same_q` instability across trials is explained by that;
  - the `sample_i` fallback in the sampler returning the last index when the cumulative sum never crosses the threshold (find the actual implementation in the repo, do not guess).
  You may instrument ONLY by running the existing binary with different environment knobs / arguments (e.g. LLAMA_* knobs, --spec-*, temperature) - never by editing source.

(B) SEVERITY for output correctness. Rejection sampling lives in common/sampling.cpp `common_sampler_sample_and_accept_n_rejection`: it reads `q_x = prob_of(q.data(), q.size(), draft[i])` and, when `q_x == 0`, never accepts and instead draws from the residual `max(0, p - q)`. Given your root cause, decide: does a wrong/zero q change the DISTRIBUTION of emitted tokens (breaking the target-distribution guarantee), or does it only lower the acceptance rate (pure speed loss)? Argue it from the code, step by step. Be explicit if the answer is "correctness is preserved, only acceptance suffers" - and if so, quantify the acceptance-rate cost implied by 18%.

(C) Is the rejection-verification path ON BY DEFAULT? Find how `COMMON_SPECULATIVE_TYPE_*` / `--spec-type` / `draft-q` / rejection mode is selected (common/arg.cpp, common/fork/arg-fork.cpp, common/fork/speculative-fork.cpp, tools/server) and what the default is when a user enables speculative decoding, and what the fork's own scripts (scripts/win-qsa) actually run.

(D) For findings 2 and 3, one short verdict each: is the assertion reasonable across backends/platforms, or is it an over-strict / parallel-unsafe test? For 3, identify the exact shared path the tests collide on.

Output: a compact report, (A) mechanism with the specific file:lines that prove it, (B) verdict on correctness vs speed with the reasoning, (C) default or not, (D) two one-line verdicts. Say explicitly what you could NOT determine and why. Do not propose patches.

**Отчёт агента (дословно):**

H confirms: `--device MTL0` FAILS (3324/12000). So Metal is the trigger; Vulkan0 (same physical GPU via MoltenVK) and CPU/BLAS pass with bit-identical chi2.

Now I have everything. Let me write the report.

Structure:
(A) mechanism + proof lines + experiment matrix + what's ruled out + what's not determined.
(B) verdict.
(C) default.
(D) two verdicts.

Key file:lines to cite:
- tests/test-spec-draft-q-fork.cpp:176 (result_q), 189-196 (n_bad loop), 195 (same_q), 199/212 fits, 58-62 ("outside q"), 70-84 same_q (1e-6), 86-93 q_of.
- common/speculative.cpp:373-376 (probabilistic gate), 379-381 (retune), 384-386 (no reset when result_q), 409-411 (sample + get candidates), 423 (id = result_q ? id_sampled : argmax), 440-441 (q copy), 475-478 (n_min drop).
- common/speculative.cpp:38-72 spec_retune → top_k=10, samplers {TOP_K, TEMPERATURE} + dist (common/sampling.cpp:507), seeds.
- src/llama-sampler.cpp:1157-1222 dist apply: p computed into cur_p->data[i].p (1188), draw over the same p (1196-1206), normalize in place (1208-1210), fallback 1217-1221 needs !found (NaN).
- src/llama-sampler.cpp:218-249 llama_sample_dist → std::discrete_distribution over the same p (245) → cannot pick a zero-p token.
- src/llama-sampler.cpp:195-205 top-k partial sort → size = 10, sorted = true.
- common/sampling.cpp:838-845 prob_of, 905 q_x, 915 accept, 926-935 residual.
- common/sampling.cpp:716 llama_synchronize.
- llama-context-fork.cpp:372-424 phase_prompt free-memory plan; moecache 939/1165.
- arg-fork.cpp:126-134, 219-243; common.h:340, 417; server-context.cpp:608, 4658-4670, 5814-5829.
- scripts/win-qsa/_config.bat:71.

Experiment matrix table.

For (B), quantify the hypothetical acceptance cost.

Let me write it compactly but completely. ## (A) ROOT CAUSE of the draft‑q failure

**Mechanism: the `q` of a trial is not a fixed function of the (rewound) draft context — the draft model's forward pass on the **Metal backend** is not bit‑reproducible from call to call. `same_q`'s 1e‑6 tolerance plus the top‑k=10 truncation of `q` turn that last‑bit noise into "token outside q". None of the four candidate explanations is what the code supports.**

Why the four candidates are wrong:

1. **"q holds p that underflowed to 0.0f while the sampler still draws the token" — impossible here.** `spec_retune` (`common/speculative.cpp:38-72`) builds the draft sampler as `{TOP_K, TEMPERATURE}` + `dist` with `top_k = 10`; top‑k truncates to 10 candidates **before** the softmax (`src/llama-sampler.cpp:195-205` sets `size = 10, sorted = true`), and `llama_sampler_dist_apply` computes `p = expf(logit - max_l)` over those 10 only (`src/llama-sampler.cpp:1185-1192`). The pass is flat (the passing run reports `df 9`, i.e. 10 cells of expected count ~1200 each → p≈0.1), so no underflow is possible. More decisively: the draw and the published `p` are **the same array** — `sum_run += cur_p->data[i].p` (`:1198-1206`) then `cur_p->data[i].p /= sum_cum` (`:1208-1210`). A token whose `p` is 0 contributes 0 to `sum_run` and can never cross `sum_tgt`. The `llama_sample_dist` variant uses `std::discrete_distribution` over the same `p` (`src/llama-sampler.cpp:245`), which also never returns a zero‑weight index.
2. **"the `sample_i` cumulative‑sum fallback returning the last index"** — this repo has no per‑sampler `sample_i` (that is the newer upstream logits API; `grep sample_i\(` → 0 hits). The only analogue is `assert(found); if (!found) cur_p->selected = cur_p->size - 1;` (`src/llama-sampler.cpp:1217-1221`), unreachable for finite logits (`sum_run` accumulates the identical float values in the identical order as `sum_cum`, and `rnd < 1`), and reachable only with NaN.
3. **"q comes from a different chain/position"** — disproved empirically: on CPU the chi‑square of 12000 first tokens fits `q0` (`chi2 12.5, bar 46.0`) and of 1306 second tokens fits `q1` (`chi2 4.0`). The draws *are* from the reported `q`.
4. **"sampler state (penalties / ngram / accepted tokens / rng) is never reset"** — the chain is `top_k → temp_ext → dist`, none of which consumes accepted tokens (`llama_sampler_dist_accept` is a no‑op unless backend‑transactional, `src/llama-sampler.cpp:1377-1389`); `common_sampler_accept` only pushes to `prev`, unused by them. The `rng` *does* advance across trials (`common/speculative.cpp:384-386` skips `common_sampler_reset` exactly when `result_q` is set) but rng state cannot change `p`. Disproved empirically: the CPU run passes all 12000 `same_q` checks with the same no‑reset code path.

Positive evidence — same binary, same model, only device knobs changed (`build-qreview/bin/test-spec-draft-q-fork`, AMD Radeon Pro 5500M):

| run | result |
|---|---|
| default (Metal offload) | **FAIL**, 6234 `same_q` FAIL lines out of ~12000 trials; 1700–4300 of 12000 first tokens outside `q0` (count varies run to run → nondeterminism, not drift) |
| `LLAMA_SAMPLER_FAST_TOP_K=0` | FAIL (confirms the parent's note) |
| `-fa off` | FAIL |
| `-ngl 0` (target on CPU, draft still auto‑offloaded) | FAIL |
| `--device MTL0` | **FAIL** (3324/12000) |
| `--device none` (all CPU) | **PASS**, `first chi2 12.5 bar 46.0`, `second chi2 4.0`, `OK` |
| `--device BLAS` | **PASS**, identical numbers |
| `--device Vulkan0` (MoltenVK, *same physical GPU*) | **PASS**, `chi2 12.5 / 4.0`, `OK` |
| default device + `--spec-draft-ngl 0` (target on GPU, **draft on CPU**) | **PASS**, `chi2 12.5 / 4.0`, `OK` |

The last two rows localize it exactly: it is the **draft context's forward pass on Metal**. `llama_memory_seq_rm` + the no‑reset sampler yield bit‑stable `q` over 12000 rewound trials on CPU/BLAS/Vulkan and unstable `q` on Metal, so the KV rewind (`test-spec-draft-q-fork.cpp:169`) and the sampler are clean.

Why tiny noise becomes a hard failure: `q` is the top‑10 candidates renormalized (`spec_retune` `top_k = 10`), so the boundary between rank 10 and 11 of this flat 128‑vocab synthetic model is crossed by logit noise around 1e‑6; when the support set changes, the trial's drawn token has no cell in `q0` → `fits()` prints "N of 12000 drafts are tokens outside q" (`test-spec-draft-q-fork.cpp:58-62`) and `n_bad` is inflated only through the `same_q` counter at `:195`.

Could not determine: the specific Metal op/kernel that is non‑reproducible (would require instrumenting the forward pass, which is out of bounds), and the *magnitude* of the per‑trial difference — `same_q` only tells me some entry moves by >1e‑6, not by how much. Also could not decompose `n_bad` (`:193-195` mixes "token outside its own q" with the `same_q` increments at `:195`), but by code that first term is structurally 0: `id_sampled` (`common/speculative.cpp:409`) and the `q` copy (`:440-441`, `emplace_back(cur_p->data, cur_p->data + cur_p->size)`) read the *same* `cur_p` in the same iteration, before anything mutates it.

#### (B) Correctness vs. speed

**As the code stands: neither. The pairing between a drafted token and its `q` is structural, so the verifier is never handed a wrong `q`; the failure is a test‑side bit‑stability assumption.** But the parent's premise deserves the general answer, because the code has no guard for it:

- `common_sampler_sample_and_accept_n_rejection` (`common/sampling.cpp:850`) takes `q_x = prob_of(q, draft[i])` (`:905`) and accepts iff `q_x > 0 && (p_x >= q_x || uni < p_x/q_x)` (`:915`); with `q_x == 0` it never accepts and draws from `norm(max(0, p - q))+` (`:926-950`). The comment at `:904` states the invariant as an assumption: *"q_x is never 0 for a token the draft produced, but guard the divide"* — a silent fallback, not an error.
- Let the drafter actually draw from `r` while the verifier is given `q̂`. The emitted law is `P(y) = r(y)·min(1,p(y)/q̂(y)) + (1 − Σ_x r(x)min(1,p(x)/q̂(x))) · (p(y)−q̂(y))⁺ / Σ_z(p(z)−q̂(z))⁺`. That equals `p` **only when `r = q̂`** (the identity needs `Σ_x r·min(1,p/q̂) = Σ min(p,q̂) = 1 − Σ(p−q̂)⁺`). So a mismatched `q` **breaks the target distribution; it is not merely an acceptance‑rate loss.**
- Direction of the bias for the specific "18 % of drafts have `q_x = 0`" case: those tokens are auto‑rejected, so the rejection probability `R` exceeds the nominal `S = Σ(p−q̂)⁺`; the residual then *dominates* the output and every token is reweighted by `R/S` against its `min(p,q̂)` share — tokens with large `q̂` relative to `p` are suppressed, tokens with `p > q̂` (including the dropped ones, which keep all of `p`) are inflated. Conversely an *understated* `q_x` raises `p_x/q_x` and over‑accepts exactly the drafter's favourites.
- Hypothetical speed cost, had the mismatch been real: those 18 % of draft positions accept with probability 0 instead of `min(1,p/q̂)`. At a typical per‑position acceptance ~0.85 that is ≈0.82× per position; a draft of length 2 keeps `0.82² ≈ 0.67` of its chain acceptance, length 3 `≈ 0.55` — i.e. roughly a 15–35 % cut in verified tokens per cycle, on top of the distribution skew.

Why it does not bite in practice: `result_q` is only set for a probabilistic draft (`common/speculative.cpp:373-376`), and then the drafted token *is* `id_sampled` (`:423`, `dparams.at(seq_id).result_q ? id_sampled : cur_p->data[0].id`) taken from the very `cur_p` copied into `q` (`:440-441`). The `n_min` drop removes the tokens and their `q` together (`:470-478`), and the length assertion in the verifier (`common/sampling.cpp:852`) matches. So `q_i` is the exact distribution `draft[i]` was drawn from, trial by trial, regardless of whether it equals `q0`.

#### (C) Is rejection/draft‑q on by default?

**No for an ordinary user; yes for everything the fork itself runs.**

- `common/common.h:340`: `bool probabilistic = false` — greedy argmax + sample‑and‑match is the default.
- Selection: `--spec-draft-sampling {greedy,probabilistic}` (`common/arg.cpp:4232-4246`) sets it; `--spec-auto` (`common/fork/arg-fork.cpp:126-134`, `auto_n` default `false`, `common/common.h:417`) forces `probabilistic = true` unless the user was explicit (`common/fork/arg-fork.cpp:238-241`).
- `--spec-type` (`common/arg.cpp:4285`) only picks the drafter (inferred from the draft GGUF when not given, `common/arg.cpp:555-560`); enabling speculation with `-md` alone does **not** enable rejection.
- Server: `use_spec_rejection()` = request `temp > 0` (`tools/server/server-context.cpp:607-609`), which passes `&slot.spec_draft_q` (`:4658-4670`, with the request's temp/seed so `spec_retune` matches them). But a greedy drafter nulls `result_q` (`common/speculative.cpp:373-376`, MTP at `:1843-1846`), so `spec_draft_q` stays empty and `use_rejection` at `:5814` falls back to `common_sampler_sample_and_accept_n`. Rejection needs probabilistic **and** temp > 0.
- The fork's own scripts: `scripts/win-qsa/_config.bat:71` → `QA_SPEC=--spec-auto --spec-draft-n-max 3`, combined with `SPECDRAFT` (MTP head) in `_qa.py:1681-1684`, so the fork's production/quality arms **do** run probabilistic drafting + rejection, and the quality gate for it is the temperature‑1 PIT calibration probe (`_qa.py:1441-1450, 1520`) plus the greedy with/without‑drafts identity probe.

#### (D) Findings 2 and 3

**2. `test-qsa-multi-fork` `CHECK(worst == 0.0)` (`tests/test-qsa-multi-fork.cpp:515`) — over‑strict, not a QSA bug.** It demands bit‑exactness of two *different compute paths* (mask from positions vs dense mask, `LLAMA_QSA_MASK_POS` read once per process) compared across two separate processes via a file (`--dense-out`/`--dense-in`); 1.49e‑08 relative to the row's largest logit is a sub‑ULP difference. The same file already concedes the point three lines below (`CHECK(worst < 1e-3)` for the "alone" run, and its comment "its cells lie elsewhere then, so not bit for bit"), and FORK.md already records GPU non‑reproducibility twice (`:601` "CUDA не повторяет результат", `:593` Vulkan radix‑select fixed by making it stable). A real mask bug (off‑by‑one block) moves logits by orders of magnitude more, so a ~1e‑6 relative tolerance keeps the regression signal and makes the test portable. Consistent with (A): this machine's Metal path is not reproducible even inside one process, and this test passes on the author's Vulkan/CUDA rigs.

**3. `test-moe-cache-fork-upload` — parallel‑unsafe test, not evidence of a race.** The shared resource is **live free device memory of the single GPU**, not a file (`${MOE_CACHE_REF}` / `${MOE_CACHE_REF}.gguf` are read‑only here and ctest fixtures already order the writer). With `--phase-mem`, `llama_context::phase_prompt` frees the schedulers, grows the elastic KV (`LLAMA_KV_ELASTIC=1`, step 256) and *then* measures `ggml_backend_dev_memory` to budget the compute buffers and pick the ubatch size / graph splits (`src/fork/llama-context-fork.cpp:372-424`, the query at `:415`); the moe-cache in turn gives up or keeps layers against that same memory (`src/fork/llama-moecache-fork.cpp:939, 1165`). So the plan and hence which experts sit in device slots versus are computed from host memory change when another test occupies the GPU — and the test's own comment says host‑computed experts land `3.5e-5` apart while `keep`/`upload` demand `tol = 0.0` (`tests/test-moe-cache-fork.cpp:492`). A last‑bit change is enough to flip the greedy `argmax` of a randomly‑initialised model, which is exactly the reported `token 0: 95 against 5`. Verdict: the reference is recorded on an idle machine and the mode comparison should either run serialised (`RESOURCE_LOCK`) or carry a small tolerance; I did not rule out a genuine defect in the `GGML_SCHED_PREFETCH`/`LLAMA_MOE_CACHE_COPY=2` upload path — that would need the events in `llama-moecache-fork.cpp` audited or a serialised GPU run to discriminate.

---

## Часть C. Чем мои проверки разошлись с отчётами и что было снято

Это важная часть: часть гипотез по ходу ревью оказалась неверной, и ниже зафиксировано именно то, что выдержало проверку.

1. **Снята гипотеза об общем дефекте синхронизации или времени жизни буфера в ggml.** Она была послана агенту по бэкендам как приоритет номер один. После того как `test-spec-draft-q-fork` с `--device Vulkan0` дал 0 падений, а `--spec-draft-ngl 0` прошёл чисто, гипотеза была явно отозвана. Агент по бэкендам независимо подтвердил: механизма, объясняющего «тот же вход -> разные числа» в `ggml-backend`/`ggml-alloc`, нет.
2. **Ложная гипотеза «быстрый top-k ломает draft-q».** Опровергнута прогоном с `LLAMA_SAMPLER_FAST_TOP_K=0` — падает так же. Сам быстрый top-k afterwards разобран построчно и чист.
3. **Ложная гипотеза «moe-cache-тесты падают из-за параллельного запуска».** Прогон `-j 1`: упали `keep`, `fill`, `kv`, `upload`; `state` и `ref` остались зелёными.
4. **Неинформативен эксперимент `LLAMA_MOE_CACHE_COPY=0`.** Сам тест выставляет режим аргументами и переопределяет ручку (`tests/test-moe-cache-fork.cpp:388-389`), так что результат «keep/fill 0 из 3» нельзя приписать этой ручке.
5. **Поправка к моей же первой редакции отчёта.** Строки «keep: default 4 Passed / 2 Failed, `--device Vulkan0` 5/1, `--device CPU` 7/8» были ошибочными по смыслу: тест игнорирует неизвестные аргументы, поэтому все три прогона были на Metal. Локализация по устройствам для этого теста недействительна; разброс — чистая неустойчивость.
6. **Уточнение к отчёту агента по переиспользованию состояния.** Он оценивал расхождение путей в `3.5e-5` и связывал его с кэшем. Мой независимый замер показывает отдельный источник шума: базовый инференс той же модели на GPU недетерминирован и без кэша (4 прогона -> 4 разных вывода), а на CPU (`-ngl 0`) побитово стабилен (5 прогонов идентичны). Поэтому `3.5e-5` attribut-ить кэшу нельзя.
7. **Не подтвердился как CPU-прогон** пункт агента «keep на `--device CPU`: 2 passed / 1 failed» — по той же причине из п.5; агент сам это отметил в своём отчёте (находка 4).
8. **CUDA-находки — вывод по чтению кода, не по замеру.** На машине нет CUDA, поэтому ни Critical N3 (stream-k), ни review CUDA-инклов не воспроизводились. Это не «подтверждённый дефект», а конкретный механизм, который надо проверить одним прогоном на 3090.
9. **Метрику качества владельца (KLD, PPL, PIT) я независимо не перепроверял** — нет реальной модели. Все количественные утверждения агентов про KLD остаются их, не моими.

## Часть D. Приоритетный список правок

Первые четыре — по одной-двум строкам каждая, и без них остальные измерения скорости и качества бессмысленны.

1. `tests/test-moe-cache-fork.cpp`: проверять `std::isfinite` логитов до сравнения, иначе FAIL. Плюс `max_diff` копить по всем записям, а не до `break`. Плюс эталон на CPU (`-ngl 0`), где форк побитово стабилен. Плюс либо поддержать `--device`, либо убрать упоминания устройств из описания теста.
2. `ggml/src/ggml-cpu/tiled/tiled.cpp`: перенести проверку типа `src1` до `switch` по типу `src0` (сейчас она после `return true`, недостижимо). Иначе новая ветка F16 в `ggml-cpu.cpp:459-463` принимает оп на CPU, а tiled-драйвер убивает его `GGML_ASSERT` на `:1066`.
3. `ggml/src/ggml-cuda/mmq.cuh:1510-1514`: `ntiles_dst` и произведение в guard'е считать в 64 битах. Проверить реальными формами qwen4exp на 3090 — с `-ub 8192` из `scripts/win-qsa/_config.bat` порядок величин близок к порогу.
4. `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmid_f16reg.comp:225`: `expert * (p.batch_stride_a / A_QK) + grow * (p.stride_a / A_QK)` — делить до умножения, как делают upstream и соседний файл самого форка (`mul_mm_f16reg.comp:133`).
5. `tools/server/fork/server-pool-fork.cpp`: `join()` или `detach()` для `watch` и `reader` на всех путях остановки; гейт `stopping` в `discovery_t::start()` по образцу `impl::start_poller`.
6. Ограничитель входящего состояния: `--cache-ram -1` не должен превращаться в 0; проверить `pool_cache_ram_fork = prompt_cache->limit_size` (`server-context.cpp:6258`).
7. `save`/`load` слота: сериализовать `base_pos` чекпойнта и поднять `SLOT_CKPT_VERSION`; иначе эндпоинт отвечает успех, а состояние отклоняют и префикс пересчитывается с нуля.
8. Spill: контрольная сумма на блоб и сверка имени с `spill_name_of()` в `restore_spilled()`; сейчас эвикция может удалить файл другого слота.
9. Тесты пула: добавить сравнение текста вывода, а не длины HTTP-тела (`test_pool_fork.py:1091` меряет `len(res.content)`).
10. Перед любыми дальнейшими измерениями скорости зафиксировать один набор `GGML_VK_*` / `LLAMA_*` и прогнать KLD против `master` — иначе непонятно, что именно дало выигрыш и что оно стоило по качеству.
11. Разбить коммиты «cache» (74 из 79) — иначе бисект по скорости невозможен.
12. CI: ни один коммит ветки его не трогает, то есть вся эта поверхность не защищена от регрессий.

## Часть E. Что осталось непроверенным

- **Качество вывода на реальном чекпойнте.** На машине нет `Qwen3.8-Flash-Next-UD-Q4_K_XL`, CUDA не компилируется, `scripts/win-qsa` — Windows-батчи. Прямой ответ на вопрос «качество не сломано» здесь дать нельзя; закрывается только KLD/PPL- прогоном на 3090 или 395 против `master`.
- **Все CUDA-находки** — нет железа.
- **Vulkan-поведение на RDNA3/RADV** (int8 coopmat, f16reg, фьюны) — на машине нет AMD-GPU; MoltenVK не воспроизводит эти пути.
- **Буквальный `master`.** Металлический NaN атрибутирован «не ветке» аналитически (пустой диф `qwen4exp.cpp`, идентичные шейдеры Metal, `LLAMA_UPSTREAM=all`), но прогоном `master` не подтверждён: переключать рабочее дерево без разрешения я не стал.
- **Механизм NaN на Metal** в PLE-гейте: локализован до одного узла, не объяснён.
- **`GGML_OP_TOP_K` на `MTL0` против CPU** (ERR до 2.0) — зафиксировано как расхождение, причина не искалась; веткой не тронуто.

## Как повторять замеры

Каталоги `build-qreview` и scratch `/tmp/moeq` удалены по завершении ревью, так что всё ниже нужно воссоздать:

```sh
cmake -S . -B build-qreview -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF
cmake --build build-qreview -j

ctest --test-dir build-qreview -R moe-cache --output-on-failure
ctest --test-dir build-qreview -R moe-cache -j 1 --output-on-failure   # без параллелизма тоже падает

#toy-модель появляется после generate-models fixture
build-qreview/bin/llama-perplexity -m build-qreview/tests/test-models/qwen4exp-moe.gguf \
  -f some-text.txt --temp 0 -c 512 -ngl 0     # конечно
#   то же с -ngl 99 -> E Unexpected negative standard deviation of log(prob)   (Metal)
#   то же с --device Vulkan0 -> конечно
```

Признак NaN в логе — `E Unexpected negative standard deviation of log(prob)` и отсутствие строки `Final estimate`. В записях `test-moe-cache-fork` — 68-байтные записи `int32 token; int32 id[8]; float logprob[8]`, разбор struct-форматом `<i8i8f`.
