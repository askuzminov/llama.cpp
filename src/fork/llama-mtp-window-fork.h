#pragma once

// the window KV of the dense MTP head of the fork's qwen4exp (LLAMA_MTP_WINDOW_KV): whether the draft context keeps only
// the last `window` positions of each sequence in the KV cache of the head, and the size of that cache. the rule of
// llama_model_qwen4exp_fork::mtp_window_kv apart from the environment and the context, so that a test checks it on
// any build (tests/test-mtp-window-fork.cpp)

#include <cstdint>

struct llama_mtp_window_kv_fork {
    bool     on      = false;
    uint32_t kv_size = 0; // cells of the cache
    uint32_t n_swa   = 0; // positions of each sequence it keeps
};

// window: LLAMA_MTP_WINDOW, 0 or less for the whole context. mode: LLAMA_MTP_WINDOW_KV, -1 with several sequences, 1
// always, 0 never
static inline llama_mtp_window_kv_fork llama_mtp_window_kv_rule_fork(
        int64_t window, int mode, uint32_t n_seq_max, bool kv_unified, uint32_t n_ubatch, uint32_t n_ctx_seq) {
    llama_mtp_window_kv_fork res;
    if (window <= 0 || mode == 0 || (mode < 0 && n_seq_max <= 1)) {
        return res;
    }
    // as llama_kv_cache_iswa sizes its SWA cache: the window of every sequence plus a ubatch. a cache of the whole
    // context or more saves nothing
    if (window >= (int64_t) n_ctx_seq) {
        return res;
    }
    const int64_t size = window*(kv_unified ? (int64_t) n_seq_max : 1) + (int64_t) n_ubatch;
    if (size >= (int64_t) n_ctx_seq) {
        return res;
    }
    res.on      = true;
    res.kv_size = (uint32_t) size;
    res.n_swa   = (uint32_t) window;
    return res;
}
