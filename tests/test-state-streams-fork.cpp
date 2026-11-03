// the state of one sequence restores into a context of another KV stream count (src/llama-kv-cache.cpp state_read_sinfo,
// fork): a context of 2 slots without kv_unified has a stream per slot, one of 1 slot or with kv_unified a single one,
// and the pool moves conversations between servers of each kind. the state holds the cells of the one stream the
// sequence lived in, which go to the stream of the sequence it is restored into. on the qwen4exp of the matrix: the
// attention cache, the recurrent state and the indexer cache that follows the layout of the attention cache.
// - the token after the restored prompt has the logits of the same token in the context the state came from
// - a whole-context state still needs the same stream count
//
// usage: test-state-streams-fork -m <dir>/qwen4exp-moe.gguf

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

struct ctx_kind {
    const char * name;
    uint32_t     n_seq_max;
    bool         unified;
};

static llama_context * make_ctx(llama_model * model, const ctx_kind & k) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = k.unified ? 256 : 256 * k.n_seq_max; // 256 a sequence, the context of the model
    cp.n_batch         = 128;
    cp.n_ubatch        = 128;
    cp.n_seq_max       = k.n_seq_max;
    cp.kv_unified      = k.unified;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.no_perf         = true;
    return llama_init_from_model(model, cp);
}

// decodes the tokens of seq from pos0 on, the logits of the last one into out
static bool decode(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos pos0, llama_seq_id seq,
                   std::vector<float> & out) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos0 + (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i]    = i + 1 == tokens.size();
    }
    batch.n_tokens = (int32_t) tokens.size();
    const bool ok = llama_decode(ctx, batch) == 0;
    if (ok) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        const float * l = llama_get_logits_ith(ctx, -1);
        out.assign(l, l + n_vocab);
    }
    llama_batch_free(batch);
    return ok;
}

static float max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float d = a.size() == b.size() ? 0.0f : INFINITY;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        d = std::fmax(d, std::fabs(a[i] - b[i]));
    }
    return d;
}

// the prompt in seq `from_seq` of a context of kind `from`, its state into seq `to_seq` of a context of kind `to`, the
// next token in both
static void move_one(llama_model * model, const ctx_kind & from, llama_seq_id from_seq, const ctx_kind & to,
                     llama_seq_id to_seq, const std::vector<llama_token> & prompt, llama_token next) {
    llama_context * a = make_ctx(model, from);
    llama_context * b = make_ctx(model, to);
    CHECK(a != nullptr && b != nullptr);
    if (a == nullptr || b == nullptr) {
        llama_free(a);
        llama_free(b);
        return;
    }

    std::vector<float> la;
    std::vector<float> lb;
    CHECK(decode(a, prompt, 0, from_seq, la));

    std::vector<uint8_t> state(llama_state_seq_get_size(a, from_seq));
    CHECK(llama_state_seq_get_data(a, state.data(), state.size(), from_seq) == state.size());
    const size_t n_read = llama_state_seq_set_data(b, state.data(), state.size(), to_seq);
    CHECK(n_read == state.size());

    CHECK(decode(a, {next}, (llama_pos) prompt.size(), from_seq, la));
    CHECK(n_read == 0 || decode(b, {next}, (llama_pos) prompt.size(), to_seq, lb));
    const float d = max_diff(la, lb);
    fprintf(stderr, "%-10s seq %d -> %-10s seq %d: state %zu bytes, max logit difference %g\n", from.name, from_seq, to.name,
            to_seq, state.size(), d);
    CHECK(d < 1e-3f);

    llama_free(a);
    llama_free(b);
}

int main(int argc, char ** argv) {
    const char * path = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            path = argv[++i];
        }
    }
    if (path == nullptr) {
        fprintf(stderr, "usage: %s -m <dir>/qwen4exp-moe.gguf\n", argv[0]);
        return 1;
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    // an indexer top-k below the cells of the prompt, so that QSA selects blocks (as test-qsa-multi-fork)
    llama_model_kv_override kvo[2];
    memset(kvo, 0, sizeof(kvo));
    kvo[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    snprintf(kvo[0].key, sizeof(kvo[0].key), "%s", "qwen4exp.attention.indexer.top_k");
    kvo[0].val_i64 = 16;
    mp.kv_overrides = kvo;

    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", path);
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> prompt;
    for (int i = 0; i < 96; ++i) {
        prompt.push_back((llama_token) ((i * 7919 + 13) % n_vocab));
    }
    const llama_token next = (llama_token) (4242 % n_vocab);

    const ctx_kind streams2 = {"2 streams", 2, false};
    const ctx_kind single   = {"1 slot",    1, false};
    const ctx_kind unified2 = {"unified 2", 2, true};

    move_one(model, streams2, 1, single,   0, prompt, next); // from the second stream into a context of one
    move_one(model, single,   0, streams2, 1, prompt, next); // into the second stream
    move_one(model, streams2, 0, unified2, 1, prompt, next);
    move_one(model, unified2, 1, streams2, 0, prompt, next);
    move_one(model, streams2, 1, streams2, 0, prompt, next); // the same count: another stream, as before

    // a whole-context state still needs the same stream count
    {
        llama_context * a = make_ctx(model, streams2);
        llama_context * b = make_ctx(model, single);
        std::vector<float> la;
        CHECK(decode(a, prompt, 0, 1, la));
        std::vector<uint8_t> state(llama_state_get_size(a));
        CHECK(llama_state_get_data(a, state.data(), state.size()) == state.size());
        CHECK(!llama_state_set_data(b, state.data(), state.size()));
        llama_free(a);
        llama_free(b);
    }

    llama_model_free(model);
    llama_backend_free();

    if (n_fail == 0) {
        fprintf(stderr, "OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
