// fork: the flatten of the deferred rollback of the gated delta net (llama_memory_recurrent::rec_flatten,
// LLAMA_GDN_LAZY). A short batch leaves records in place of the snapshot planes, and a state save turns the records into
// the planes the plain net would have written. After the same batches and the same rollback the saved state of a
// context with records must equal, byte for byte, that of a context without the deferred rollback (LLAMA_GDN_LAZY=0),
// and so must the logits of the next token. The tests of test-llama-context cannot see a wrong flatten: the save
// flattens the source context in place, so the source and the restored copy agree whatever the flatten wrote.
// Since 12.11 the info rows of the replay are allocated with the RS buffers: a replay allocated them on each flatten,
// and a refused allocation of 256 bytes aborted the server on a full card (the 3090 with the CUDA guard of the free
// memory). The knob is read when a context makes its memory, so both contexts live in one process.
//
// usage: test-gdn-flatten-fork -m <dir>/qwen4exp-moe.gguf [-dev <name>]

#include "llama.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

static std::mutex  g_log_mutex;
static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void * /*user_data*/) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log += text;
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static constexpr uint32_t N_RS_SEQ   = 8;
static constexpr int      N_PROMPT   = 24; // one long batch: the plain net
static constexpr int      N_SHORT    = 5;  // one short batch: records with the deferred rollback

struct result {
    bool                 ok      = false;
    bool                 records = false; // the context made records (the deferred rollback was on)
    std::vector<uint8_t> state;
    std::vector<float>   logits;
};

static bool decode(llama_context * ctx, const std::vector<llama_token> & toks, int pos0) {
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.token   [i]    = toks[i];
        batch.pos     [i]    = pos0 + (llama_pos) i;
        batch.n_seq_id[i]    = 1;
        batch.seq_id  [i][0] = 0;
        batch.logits  [i]    = i + 1 == toks.size();
    }
    batch.n_tokens = (int32_t) toks.size();
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// rollback: the positions taken back after the short batch before the save; 0 reads plane 0 (the base and all the
// records), 2 plane 2 (the base and the records but the last two)
static result run(llama_model * model, const char * lazy, const std::vector<llama_token> & toks, int rollback) {
    set_env("LLAMA_GDN_LAZY", lazy);
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_log.clear();
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 256;
    cp.n_batch   = 64;
    cp.n_ubatch  = 64;
    cp.n_seq_max = 1;
    cp.n_rs_seq  = N_RS_SEQ;
    cp.no_perf   = true;

    result r;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        fprintf(stderr, "cannot create the context (LLAMA_GDN_LAZY=%s)\n", lazy);
        return r;
    }
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        r.records = g_log.find("deferred rollback of the gated delta net: records") != std::string::npos;
    }

    const std::vector<llama_token> prompt(toks.begin(), toks.begin() + N_PROMPT);
    const std::vector<llama_token> burst(toks.begin() + N_PROMPT, toks.begin() + N_PROMPT + N_SHORT);
    const int pos_back = N_PROMPT + N_SHORT - rollback;

    bool ok = decode(ctx, prompt, 0) && decode(ctx, burst, N_PROMPT);
    // the rollback of a short batch: a snapshot plane, or the records of the deferred rollback
    if (rollback > 0) {
        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx), 0, pos_back, -1);
    }

    // the save flattens the records into planes first
    if (ok) {
        const size_t size = llama_state_seq_get_size(ctx, 0);
        r.state.resize(size);
        ok = size > 0 && llama_state_seq_get_data(ctx, r.state.data(), r.state.size(), 0) == size;
    }

    // the next token reads the state the flatten left
    ok = ok && decode(ctx, { toks[pos_back % (int) toks.size()] }, pos_back);
    if (ok) {
        const int     n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * logits  = llama_get_logits_ith(ctx, -1);
        r.logits.assign(logits, logits + n_vocab);
    }

    llama_free(ctx);
    r.ok = ok;
    if (!ok) {
        fprintf(stderr, "the batches, the rollback or the save failed (LLAMA_GDN_LAZY=%s)\n", lazy);
    }
    return r;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * dev_name   = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((strcmp(argv[i], "-dev") == 0 || strcmp(argv[i], "--device") == 0) && i + 1 < argc) {
            dev_name = argv[++i];
        } else {
            model_path = nullptr;
            break;
        }
    }
    if (model_path == nullptr) {
        fprintf(stderr, "usage: %s -m <qwen4exp-moe.gguf> [-dev <name>]\n", argv[0]);
        return 1;
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    if (dev_name != nullptr) {
        devs[0] = ggml_backend_dev_by_name(dev_name);
        if (devs[0] == nullptr) {
            fprintf(stderr, "%s: no device %s\n", argv[0], dev_name);
            llama_backend_free();
            return 1;
        }
        mp.devices      = devs;
        mp.n_gpu_layers = 99;
    } else {
        mp.n_gpu_layers = 0;
    }

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", model_path);
        llama_backend_free();
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> toks(N_PROMPT + N_SHORT + 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        toks[i] = (llama_token) (3 + (i*37 + 11) % (size_t) (n_vocab - 3));
    }

    for (const int rollback : { 2, 0 }) {
        const result lazy  = run(model, "1", toks, rollback);
        const result plain = run(model, "0", toks, rollback);

        CHECK(lazy.ok && plain.ok);
        CHECK(lazy.records);   // else the test compares two plain contexts
        CHECK(!plain.records);
        CHECK(lazy.state.size() == plain.state.size());
        size_t n_diff = 0;
        for (size_t i = 0; i < std::min(lazy.state.size(), plain.state.size()); ++i) {
            n_diff += lazy.state[i] != plain.state[i] ? 1 : 0;
        }
        if (n_diff > 0) {
            fprintf(stderr, "rollback %d: the saved states differ in %zu of %zu bytes\n", rollback, n_diff, lazy.state.size());
        }
        CHECK(n_diff == 0);
        CHECK(lazy.logits.size() == plain.logits.size() && !lazy.logits.empty());
        float max_diff = 0.0f;
        for (size_t i = 0; i < std::min(lazy.logits.size(), plain.logits.size()); ++i) {
            max_diff = std::max(max_diff, std::fabs(lazy.logits[i] - plain.logits[i]));
        }
        if (max_diff > 0.0f) {
            fprintf(stderr, "rollback %d: the logits of the next token differ by up to %g\n", rollback, max_diff);
        }
        CHECK(max_diff == 0.0f);
        printf("%s: rollback %d, state %zu bytes, %zu bytes apart, logits %g apart\n", argv[0], rollback,
               lazy.state.size(), n_diff, max_diff);
    }

    llama_model_free(model);
    llama_backend_free();

    printf("%s: %s, %s\n", argv[0], dev_name ? dev_name : "CPU", g_fail == 0 ? "the flatten matches the plain net" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
