// fork: a partial rollback of the recurrent state after a decode that held sequences of unequal length
// (llama_memory_recurrent::find_slot, seq_rm). The memory answers the rollback of a sequence from the snapshots of its
// last batch (or the records of the deferred rollback, LLAMA_GDN_LAZY). A decode with several sequences runs as several
// ubatches, and a later one can move the cell of a sequence that is done, or copy it along: its snapshots are gone
// then. The rollback must be refused, or it must leave the state the sequence had alone: until 16.11 it was accepted
// and read the plane of another sequence (the verify of one server slot in a decode with the prompt pieces of two
// others). The state is compared, not the logits: the generated model moves them too little.
//
// usage: test-rs-rollback-fork -m <dir>/qwen4exp-moe.gguf

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static constexpr uint32_t N_RS_SEQ = 7;
static constexpr int      N_SEQ    = 3;
static constexpr int      N_PROMPT = 16; // a long batch
static constexpr int      N_SHORT  = 4;  // a short batch: it leaves snapshots
static constexpr int      N_BACK   = 2;  // positions taken back

// the tokens of a sequence do not depend on what runs next to it
static std::vector<llama_token> tokens_of(int seq, int n) {
    std::mt19937 rng(1000 + seq);
    std::vector<llama_token> res(n);
    for (auto & t : res) {
        t = 3 + (llama_token) (rng() % 100);
    }
    return res;
}

struct piece {
    int seq;
    int n;
};

// one decode: the pieces in the order given, each sequence goes on from its position
static bool decode(llama_context * ctx, const std::vector<piece> & pieces, std::vector<int> & pos) {
    int n_all = 0;
    for (const piece & p : pieces) {
        n_all += p.n;
    }
    llama_batch batch = llama_batch_init(n_all, 0, 1);
    int i = 0;
    for (const piece & p : pieces) {
        const std::vector<llama_token> toks = tokens_of(p.seq, pos[p.seq] + p.n);
        for (int k = 0; k < p.n; ++k, ++i) {
            batch.token   [i]    = toks[pos[p.seq] + k];
            batch.pos     [i]    = pos[p.seq] + k;
            batch.n_seq_id[i]    = 1;
            batch.seq_id  [i][0] = p.seq;
            batch.logits  [i]    = k + 1 == p.n;
        }
        pos[p.seq] += p.n;
    }
    batch.n_tokens = n_all;
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static llama_context * make_ctx(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 1024;
    cp.n_batch    = 256;
    cp.n_ubatch   = 256;
    cp.n_seq_max  = N_SEQ;
    cp.kv_unified = true;
    cp.n_rs_seq   = N_RS_SEQ;
    cp.no_perf    = true;
    return llama_init_from_model(model, cp);
}

// the part of the state that cannot roll back by itself, a rollback that waits applied
static std::vector<uint8_t> state_of(llama_context * ctx, int seq) {
    std::vector<uint8_t> res(llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    if (res.empty() || llama_state_seq_get_data_ext(ctx, res.data(), res.size(), seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != res.size()) {
        res.clear();
    }
    return res;
}

// the words that differ read as floats: the states of one sequence computed in two batch layouts differ by rounding
static bool same_state(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    if (a.empty() || a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i + 4 <= a.size(); i += 4) {
        if (memcmp(a.data() + i, b.data() + i, 4) == 0) {
            continue;
        }
        float x;
        float y;
        memcpy(&x, a.data() + i, 4);
        memcpy(&y, b.data() + i, 4);
        if (!std::isfinite(x) || !std::isfinite(y) || std::fabs(x - y) > 1e-3f*std::max(1.0f, std::max(std::fabs(x), std::fabs(y)))) {
            return false;
        }
    }
    return true;
}

// the sequence `a` alone: its prompt, a short batch, the verify batch, the rollback
static std::vector<uint8_t> run_alone(llama_model * model, int a) {
    llama_context * ctx = make_ctx(model);
    if (ctx == nullptr) {
        return {};
    }
    std::vector<int> pos(N_SEQ, 0);
    bool ok = decode(ctx, {{a, N_PROMPT}}, pos) && decode(ctx, {{a, N_SHORT}}, pos) && decode(ctx, {{a, N_SHORT}}, pos);
    ok = ok && llama_memory_seq_rm(llama_get_memory(ctx), a, pos[a] - N_BACK, -1);
    CHECK(ok); // one sequence: the rollback of a short batch works
    std::vector<uint8_t> res = ok ? state_of(ctx, a) : std::vector<uint8_t>();
    llama_free(ctx);
    return res;
}

// 0: the rollback was refused, 1: accepted and right, -1: accepted and wrong
static int run_joint(llama_model * model, int a, const std::vector<uint8_t> & ref) {
    llama_context * ctx = make_ctx(model);
    if (ctx == nullptr) {
        return -1;
    }
    std::vector<int> pos(N_SEQ, 0);
    bool ok = true;
    // every sequence in its cell, each with snapshots of a short batch
    for (int s = 0; s < N_SEQ && ok; ++s) {
        ok = decode(ctx, {{s, N_PROMPT}}, pos) && decode(ctx, {{s, N_SHORT}}, pos);
    }
    // the verify of `a` with the prompt pieces of the two others, the sequences in the order of their cells
    std::vector<piece> pieces;
    for (int s = 0; s < N_SEQ; ++s) {
        pieces.push_back({s, s == a ? N_SHORT : 12 + 8*((s + 1) % N_SEQ)});
    }
    ok = ok && decode(ctx, pieces, pos);
    CHECK(ok);

    int res = -1;
    if (ok) {
        if (!llama_memory_seq_rm(llama_get_memory(ctx), a, pos[a] - N_BACK, -1)) {
            res = 0;
        } else {
            res = same_state(state_of(ctx, a), ref) ? 1 : -1;
        }
    }
    llama_free(ctx);
    return res;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        }
    }
    if (model_path == nullptr) {
        fprintf(stderr, "usage: %s -m <qwen4exp-moe.gguf>\n", argv[0]);
        return 1;
    }

    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", model_path);
        llama_backend_free();
        return 1;
    }

    // the knob is read when a context makes its memory
    for (const char * lazy : { "0", "1" }) {
        set_env("LLAMA_GDN_LAZY", lazy);
        int n_right = 0;
        for (int a = 0; a < N_SEQ; ++a) {
            const std::vector<uint8_t> ref = run_alone(model, a);
            CHECK(!ref.empty());
            const int res = run_joint(model, a, ref);
            printf("LLAMA_GDN_LAZY=%s, the verify of sequence %d next to two prompts: the rollback %s\n", lazy, a,
                    res == 0 ? "refused" : res == 1 ? "accepted, the state as alone" : "accepted, ANOTHER STATE");
            CHECK(res >= 0);
            n_right += res == 1;
        }
        // not every layout loses the snapshots: the first cell is never moved or copied along
        CHECK(n_right >= 1 || strcmp(lazy, "1") == 0);
    }

    llama_model_free(model);
    llama_backend_free();

    if (g_fail > 0) {
        fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
