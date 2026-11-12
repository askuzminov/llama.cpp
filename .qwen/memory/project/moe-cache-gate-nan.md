---
description: "Гейт качества test-moe-cache-fork пропускает полностью NaN-вывод; NaN рождается в Metal-пути qwen4exp, а не в кэше экспертов"
type: project
category: common_pitfall
keywords:
  - "test-moe-cache-fork"
  - "NaN-эталон"
  - "Metal qwen4exp"
  - "moe-cache-review"
  - "llama-perplexity"
usage_scenarios:
  - "Ревью или отладка гейтов качества ветки cache"
  - "Падения test-moe-cache-fork на_macOS"
  - "Оценка качества вывода qwen4exp на GPU"
---

Гейт `tests/test-moe-cache-fork.cpp` не является проверкой качества: его эталон (`--out`, fixture `moe-cache-ref` генерирует его на машине прогона) в большинстве прогонов на macOS состоит из NaN — 1536 из 1536 logprob, токен повторён 192 раза. Сравнение это не замечает: `max_diff = std::max(max_diff, fabs(a-b))` при NaN возвращает старый `max_diff`, а порядок топ-8 при всех NaN детерминирован, поэтому NaN-vs-NaN проходит бит-в-бит. Плюс `break` при первом несовпадении токенов стоит до накопления `max_diff` — сообщение «logprobs 0 apart» означает «ничего не измерено».

**Why:** NaN приходит не из кэша экспертов. На toy-модели `test-models/qwen4exp-moe.gguf` со случайно сгенерированными весами и с выключенными по умолчанию `--moe-cache`/`--phase-mem`: CPU `-ngl 0` даёт PPL 128.4806, `--device Vulkan0 -ngl 99` даёт 128.4648 (совпадение с CPU до 5 знаков), а Metal — NaN в 4/4 прогонов. `LLAMA_UPSTREAM=all`, `LLAMA_UPSTREAM=qwen4exp`, `LLAMA_HC_F16=0`, `LLAMA_KV_ELASTIC=0`, `-fa off`, `LLAMA_MOE_DOWN_F16=0` NaN не убирают. `ggml-metal` в ветке тронут на 5 строк, ggml-ядро — на +705.

**How to apply:** Не трактовать зелёные режимы keep/fill/kv/upload как подтверждение корректности весов кэша. Каталог сборки `build-qreview` и scratch `/tmp/moeq` удалены 10.11 по просьбе пользователя — замеры надо повторять с нуля, путей из них не осталось. Проверка конечности (`std::isfinite`) перед сравнением и эталон на CPU (`-ngl 0`, там форк бит-стобилен) — то, что делает гейт рабочим. Сам источник NaN на этой машине нераскрыт: master не пересобран, реальной модели Qwen3.8-Flash-Next на машине нет, CUDA не компилируется — прямую проверку качества вывода на реальном чекпойнте здесь провести нельзя.
