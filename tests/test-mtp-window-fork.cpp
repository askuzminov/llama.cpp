// the window KV of the dense MTP head (LLAMA_MTP_WINDOW_KV, src/fork/llama-mtp-window-fork.h): when the draft context
// keeps only the last positions of each sequence in the KV cache of the head, and how many cells that cache gets. the
// rule is header-only, so this runs on a Windows DLL build too

#include "fork/llama-mtp-window-fork.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>

static int n_fail = 0;

static void check(bool ok, const char * what, int line) {
    if (!ok) {
        fprintf(stderr, "line %d: %s\n", line, what);
        n_fail++;
    }
}

#define CHECK(x) check((x), #x, __LINE__)

static bool on(int64_t window, int mode, uint32_t n_seq_max, bool kv_unified, uint32_t n_ubatch, uint32_t n_ctx_seq) {
    return llama_mtp_window_kv_rule_fork(window, mode, n_seq_max, kv_unified, n_ubatch, n_ctx_seq).on;
}

int main() {
    const uint32_t n_ctx = 262144;

    // no window (LLAMA_MTP_WINDOW=0 or below): the whole context in every mode
    for (int mode : { -1, 0, 1 }) {
        CHECK(!on(0,  mode, 4, true, 1024, n_ctx));
        CHECK(!on(-1, mode, 4, true, 1024, n_ctx));
    }

    // LLAMA_MTP_WINDOW_KV=0: never
    CHECK(!on(16384, 0, 4, true,  1024, n_ctx));
    CHECK(!on(16384, 0, 1, false, 1024, n_ctx));

    // -1 (the default): only with several sequences, one slot keeps the whole cache (its window is a view of the last
    // cells)
    CHECK(!on(16384, -1, 1, true,  1024, n_ctx));
    CHECK(!on(16384, -1, 1, false, 1024, n_ctx));

    // several slots of a unified cache (par4 of the harness): the window of every sequence plus a ubatch
    {
        const auto r = llama_mtp_window_kv_rule_fork(16384, -1, 4, true, 1024, n_ctx);
        CHECK(r.on);
        CHECK(r.kv_size == 16384*4 + 1024);
        CHECK(r.n_swa == 16384);
    }

    // a stream per sequence: each stream holds the window of one
    {
        const auto r = llama_mtp_window_kv_rule_fork(16384, -1, 4, false, 1024, n_ctx/4);
        CHECK(r.on);
        CHECK(r.kv_size == 16384 + 1024);
        CHECK(r.n_swa == 16384);
    }

    // 1: one slot too (the mtpkv arm)
    {
        const auto r = llama_mtp_window_kv_rule_fork(16384, 1, 1, true, 512, n_ctx);
        CHECK(r.on);
        CHECK(r.kv_size == 16384 + 512);
        CHECK(r.n_swa == 16384);
    }

    // the Mac check of the prompt-cache signature: window 64, ubatch 64, 256 cells -> 128
    {
        const auto r = llama_mtp_window_kv_rule_fork(64, 1, 1, true, 64, 256);
        CHECK(r.on);
        CHECK(r.kv_size == 128);
        CHECK(r.n_swa == 64);
    }

    // a cache of the size of the context or more saves nothing: the boundary
    CHECK(!on(64, 1, 1, true, 64, 128));                 // 128 cells for 128
    CHECK( on(64, 1, 1, true, 63, 128));                 // 127
    CHECK(!on(16384, -1, 16, true, 1024, n_ctx));        // 16 slots: 263168 cells for 262144
    CHECK( on(16384, -1, 15, true, 1024, n_ctx));        // 15: 246784
    CHECK(!on(n_ctx, 1, 1, true, 0, n_ctx));             // a window of the whole context
    CHECK(!on(int64_t(1) << 40, 1, 64, true, 512, n_ctx)); // no overflow of window x slots

    // a window from the environment past the context: no window, and never a truncated one. the formula without the
    // bound overflowed window x slots (2^62 x 4 wraps to 0: a cache of n_ubatch cells) and cast the window to the
    // n_swa of a standard SWA cache, 0 for 2^62, where the head would see its own cell only
    CHECK(!on(INT64_MAX, 1, 4, true, 512, n_ctx));
    CHECK(!on(int64_t(1) << 62, 1, 4, true, 512, n_ctx));
    CHECK(!on(int64_t(1) << 32, 1, 1, true, 512, UINT32_MAX));

    // whatever the inputs, a window that is on is the window asked for, below the context, in a cache below the context
    {
        const int64_t  windows[] = { 1, 63, 64, 16384, 262143, 262144, int64_t(1) << 31, int64_t(1) << 32,
                                     (int64_t(1) << 32) + 64, int64_t(1) << 40, int64_t(1) << 62, INT64_MAX };
        const uint32_t seqs   [] = { 1, 2, 4, 64, 256 };
        const uint32_t ubatch [] = { 0, 512, 4096 };
        const uint32_t ctxs   [] = { 256, n_ctx, UINT32_MAX };
        int n_on  = 0;
        int n_bad = 0;
        for (int64_t w : windows) {
            for (uint32_t s : seqs) {
                for (uint32_t u : ubatch) {
                    for (uint32_t c : ctxs) {
                        for (bool unified : { false, true }) {
                            const auto r = llama_mtp_window_kv_rule_fork(w, 1, s, unified, u, c);
                            if (!r.on) {
                                continue;
                            }
                            n_on++;
                            n_bad += !(r.n_swa >= 1 && (int64_t) r.n_swa == w && r.n_swa < c && r.kv_size < c && r.kv_size >= r.n_swa);
                        }
                    }
                }
            }
        }
        CHECK(n_on > 0);
        CHECK(n_bad == 0);
    }

    // a rule that says no sets nothing: mtp_window_kv then leaves the sizes of create_memory alone
    {
        const auto r = llama_mtp_window_kv_rule_fork(16384, -1, 1, true, 1024, n_ctx);
        CHECK(r.kv_size == 0 && r.n_swa == 0);
    }

    if (n_fail > 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }

    printf("OK\n");

    return 0;
}
