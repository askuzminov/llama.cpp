// fork: the elastic KV cache with several streams (LLAMA_KV_ELASTIC, src/fork/llama-kv-elastic-fork.cpp): a stream per
// slot without --kv-unified, each mapping the cells it uses. The reference run (--out) keeps the static cache
// (LLAMA_KV_ELASTIC=0) and writes the logits of a scenario; each mode runs the same scenario on an elastic cache emulated
// in ordinary buffers and must give the same logits bit for bit:
// - zero: reads past the capacity of a stream give zeros, as on a Vulkan device with residencyNonResidentStrict, so a
//   short stream keeps its few cells while its joint ubatches with the long one read many more
// - map: they do not (CUDA VMM): the streams of a joint ubatch map what it reads. the cells past a capacity hold NaN
//   (LLAMA_KV_ELASTIC_POISON=1), so a read of them shows in the logits
// - reclaim: the memory stops at 3 MiB (LLAMA_KV_ELASTIC_TEST_MIB, ~2560 cells of 1.2 KiB together): stream 0 grows to
//   ~1800 cells and is cleared, stream 1 then fits a long prompt only with the memory stream 0 still maps (no shrink
//   without --phase-mem), which it takes back. no reference: every decode succeeds
// the scenario on the qwen4exp of the matrix (attention, gated delta net, indexer cache): a long and a short prompt in two
// streams, joint decode steps, a rollback of the drafts of both (n_rs_seq), the state of the long one restored into
// stream 1 of another context, a copy of stream 0 to stream 1 (seq_cp across streams), a K-shift of stream 0
//
// on the CPU the attention skips masked cells, so the NaN past a capacity shows only where a kernel reads them: --ngl 99
// on a GPU (the Vulkan of this Mac emulates the elastic buffers too)
//
// usage: test-kv-elastic-streams-fork -m <dir>/qwen4exp-moe.gguf [--ngl N] (--out <file> | --ref <file> --mode zero|map | --mode reclaim)

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static std::mutex  g_log_mutex;
static std::string g_log;
static size_t      g_mark = 0; // the log up to the joint steps of the two prompts and their rollbacks

static void log_cb(ggml_log_level level, const char * text, void *) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log += text;
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static constexpr uint32_t N_CTX_SEQ = 2048; // cells of a stream
static constexpr uint32_t N_RS_SEQ  = 4;    // recurrent snapshots: the drafts a rollback takes back

static llama_context * make_ctx(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 2*N_CTX_SEQ;
    cp.n_batch         = 512;
    cp.n_ubatch        = 512;
    cp.n_seq_max       = 2;
    cp.kv_unified      = false;
    cp.n_rs_seq        = N_RS_SEQ;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.no_perf         = true;
    return llama_init_from_model(model, cp);
}

struct tok {
    llama_seq_id seq;
    llama_pos    pos;
    llama_token  t;
};

// the logits of every decoded token, in order
static std::vector<float> g_logits;
static int                g_n_vocab = 0;

static bool decode(llama_context * ctx, const std::vector<tok> & toks, bool take) {
    for (size_t i0 = 0; i0 < toks.size(); i0 += 512) {
        const size_t n = std::min<size_t>(512, toks.size() - i0);
        llama_batch b = llama_batch_init((int32_t) n, 0, 1);
        for (size_t i = 0; i < n; ++i) {
            b.token[i]     = toks[i0 + i].t;
            b.pos[i]       = toks[i0 + i].pos;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = toks[i0 + i].seq;
            b.logits[i]    = take;
        }
        b.n_tokens = (int32_t) n;
        const bool ok = llama_decode(ctx, b) == 0;
        if (ok && take) {
            for (size_t i = 0; i < n; ++i) {
                const float * l = llama_get_logits_ith(ctx, (int32_t) i);
                g_logits.insert(g_logits.end(), l, l + g_n_vocab);
            }
        }
        llama_batch_free(b);
        if (!ok) {
            return false;
        }
    }
    return true;
}

static llama_token tk(int i) {
    return (llama_token) ((i * 7919 + 13) % g_n_vocab);
}

static void scenario(llama_model * model) {
    llama_context * ctx = make_ctx(model);
    CHECK(ctx != nullptr);
    if (ctx == nullptr) {
        return;
    }
    llama_memory_t mem = llama_get_memory(ctx);

    // a long prompt in stream 0, a short one in stream 1
    const llama_pos n_a = 1300;
    const llama_pos n_b = 40;
    std::vector<tok> pa;
    std::vector<tok> pb;
    for (llama_pos i = 0; i < n_a; ++i) {
        pa.push_back({ 0, i, tk(i) });
    }
    for (llama_pos i = 0; i < n_b; ++i) {
        pb.push_back({ 1, i, tk(5000 + i) });
    }
    CHECK(decode(ctx, pa, false));
    CHECK(decode(ctx, pb, false));

    // joint steps: each ubatch reads the n_kv of the long stream in both
    llama_pos pos_a = n_a;
    llama_pos pos_b = n_b;
    for (int k = 0; k < 6; ++k) {
        CHECK(decode(ctx, { { 0, pos_a, tk(9000 + k) }, { 1, pos_b, tk(9500 + k) } }, true));
        pos_a++;
        pos_b++;
    }

    // drafts of both, then a rollback of them as after a rejected speculation
    CHECK(decode(ctx, { { 0, pos_a, tk(1) }, { 0, pos_a + 1, tk(2) }, { 0, pos_a + 2, tk(3) },
                        { 1, pos_b, tk(4) }, { 1, pos_b + 1, tk(5) } }, true));
    CHECK(llama_memory_seq_rm(mem, 0, pos_a + 1, -1));
    CHECK(llama_memory_seq_rm(mem, 1, pos_b + 1, -1));
    pos_a++;
    pos_b++;
    for (int k = 0; k < 3; ++k) {
        CHECK(decode(ctx, { { 0, pos_a, tk(9800 + k) }, { 1, pos_b, tk(9900 + k) } }, true));
        pos_a++;
        pos_b++;
    }

    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_mark = g_log.size();
    }

    // the state of the long sequence into stream 1 of another context: its next token there and here
    {
        std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
        CHECK(llama_state_seq_get_data(ctx, state.data(), state.size(), 0) == state.size());
        llama_context * other = make_ctx(model);
        CHECK(other != nullptr);
        if (other != nullptr) {
            CHECK(llama_state_seq_set_data(other, state.data(), state.size(), 1) == state.size());
            CHECK(decode(other, { { 1, pos_a, tk(7) } }, true));
            llama_free(other);
        }
        CHECK(decode(ctx, { { 0, pos_a, tk(7) } }, true));
        pos_a++;
    }

    // stream 0 copied to stream 1: the same next token on both
    CHECK(llama_memory_seq_rm(mem, 1, -1, -1));
    llama_memory_seq_cp(mem, 0, 1, -1, -1);
    CHECK(decode(ctx, { { 0, pos_a, tk(8) }, { 1, pos_a, tk(8) } }, true));
    pos_a++;

    // a K-shift of the positions of stream 0, applied before the next decode
    if (llama_memory_can_shift(mem)) {
        llama_memory_seq_add(mem, 0, 0, -1, 10);
        pos_a += 10;
        CHECK(decode(ctx, { { 0, pos_a, tk(9) }, { 1, pos_a - 10, tk(9) } }, true));
    }

    llama_free(ctx);
}

static void scenario_reclaim(llama_model * model) {
    llama_context * ctx = make_ctx(model);
    CHECK(ctx != nullptr);
    if (ctx == nullptr) {
        return;
    }
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<tok> pa;
    std::vector<tok> pb;
    for (llama_pos i = 0; i < 1300; ++i) {
        pa.push_back({ 0, i, tk(i) });
        pb.push_back({ 1, i, tk(3000 + i) });
    }
    CHECK(decode(ctx, pa, false));
    CHECK(llama_memory_seq_rm(mem, 0, -1, -1));
    CHECK(decode(ctx, pb, false));
    CHECK(decode(ctx, { { 1, 1300, tk(1) } }, true));

    llama_free(ctx);
}

int main(int argc, char ** argv) {
    const char * path = nullptr;
    const char * out  = nullptr;
    const char * ref  = nullptr;
    std::string  mode;
    int          ngl  = 0;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            path = argv[++i];
        } else if (strcmp(argv[i], "--ngl") == 0) {
            ngl = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0) {
            out = argv[++i];
        } else if (strcmp(argv[i], "--ref") == 0) {
            ref = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0) {
            mode = argv[++i];
        }
    }
    const bool reclaim = mode == "reclaim" && out == nullptr && ref == nullptr;
    if (path == nullptr || (!reclaim && ((out == nullptr) == (ref == nullptr) || (ref != nullptr && mode != "zero" && mode != "map")))) {
        fprintf(stderr, "usage: %s -m <dir>/qwen4exp-moe.gguf [--ngl N] (--out <file> | --ref <file> --mode zero|map | --mode reclaim)\n", argv[0]);
        return 1;
    }

    // the knobs are read once per process
    if (reclaim) {
        set_env("LLAMA_KV_ELASTIC", "1");
        set_env("LLAMA_KV_ELASTIC_STEP", "256");
        set_env("LLAMA_KV_ELASTIC_TEST_MIB", "3");
    } else if (out != nullptr) {
        set_env("LLAMA_KV_ELASTIC", "0");
    } else {
        set_env("LLAMA_KV_ELASTIC", "1");
        set_env("LLAMA_KV_ELASTIC_STEP", "256");
        set_env("LLAMA_KV_ELASTIC_ZERO_READS", mode == "zero" ? "1" : "0");
        set_env("LLAMA_KV_ELASTIC_POISON", "1");
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    // an indexer top-k below the cells of the prompts, so that QSA selects blocks; a context past the 256 of the matrix
    llama_model_kv_override kvo[3];
    memset(kvo, 0, sizeof(kvo));
    kvo[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    snprintf(kvo[0].key, sizeof(kvo[0].key), "%s", "qwen4exp.attention.indexer.top_k");
    kvo[0].val_i64 = 16;
    kvo[1].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    snprintf(kvo[1].key, sizeof(kvo[1].key), "%s", "qwen4exp.context_length");
    kvo[1].val_i64 = 16384;
    mp.kv_overrides = kvo;

    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", path);
        return 1;
    }
    g_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    if (reclaim) {
        scenario_reclaim(model);
        std::lock_guard<std::mutex> lock(g_log_mutex);
        CHECK(g_log.find("the other streams give back") != std::string::npos);
    } else {
        scenario(model);
    }

    if (reclaim) {
        // checked above
    } else if (out != nullptr) {
        FILE * f = fopen(out, "wb");
        CHECK(f != nullptr);
        if (f != nullptr) {
            CHECK(fwrite(g_logits.data(), sizeof(float), g_logits.size(), f) == g_logits.size());
            fclose(f);
        }
    } else {
        std::vector<float> want;
        FILE * f = fopen(ref, "rb");
        CHECK(f != nullptr);
        if (f != nullptr) {
            float buf[4096];
            size_t n;
            while ((n = fread(buf, sizeof(float), 4096, f)) > 0) {
                want.insert(want.end(), buf, buf + n);
            }
            fclose(f);
        }
        CHECK(want.size() == g_logits.size());
        size_t n_diff = 0;
        for (size_t i = 0; i < std::min(want.size(), g_logits.size()); ++i) {
            n_diff += memcmp(&want[i], &g_logits[i], sizeof(float)) != 0 ? 1 : 0;
        }
        fprintf(stderr, "mode %s: %zu logits, %zu differ from the static cache\n", mode.c_str(), g_logits.size(), n_diff);
        CHECK(n_diff == 0);

        // the path ran: an elastic cache of 2 streams in the mode asked for, and no cell in use lost its memory
        std::lock_guard<std::mutex> lock(g_log_mutex);
        for (size_t at = 0; at < g_log.size(); ) {
            const size_t end  = g_log.find('\n', at);
            const std::string line = g_log.substr(at, end == std::string::npos ? std::string::npos : end - at);
            if (line.find("offloaded") != std::string::npos || line.find("streams, each maps") != std::string::npos ||
                line.find("MiB per 1024 cells") != std::string::npos ||
                line.find("emulated") != std::string::npos) {
                fprintf(stderr, "%s\n", line.c_str());
            }
            at = end == std::string::npos ? g_log.size() : end + 1;
        }
        CHECK(g_log.find("kv_elastic: 2 streams, each maps its own cells") != std::string::npos);
        CHECK(g_log.find(mode == "zero" ? "read as zeros" : "the streams of a ubatch map the cells it reads") != std::string::npos);
        CHECK(g_log.find("its data is lost") == std::string::npos);
        CHECK(g_log.find("the ubatch reads") == std::string::npos);
        // the capacity of stream 1 through the joint steps: the n_kv of their ubatches is 1536 (1300 to 1310 cells of
        // stream 0), which stream 1 maps in the map mode only
        uint32_t cap1 = 0;
        const std::string head = g_log.substr(0, g_mark);
        // the first capacity of every stream, then the changes of stream 1 or of all of them
        const size_t first = head.find(" cells mapped in each of 2 streams");
        if (first != std::string::npos) {
            const size_t at = head.rfind("kv_elastic: ", first);
            unsigned a = 0;
            unsigned b = 0;
            if (at != std::string::npos && sscanf(head.c_str() + at, "kv_elastic: %u of %u cells mapped", &a, &b) == 2) {
                cap1 = a;
            }
        }
        for (const char * key : { "kv_elastic: stream 1 ", "kv_elastic: every stream " }) {
            for (size_t at = head.find(key); at != std::string::npos; at = head.find(key, at + 1)) {
                unsigned a = 0;
                unsigned b = 0;
                if (sscanf(head.c_str() + at + strlen(key), "%u -> %u cells", &a, &b) == 2) {
                    cap1 = std::max<uint32_t>(cap1, b);
                }
            }
        }
        fprintf(stderr, "mode %s: stream 1 mapped at most %u cells through the joint steps\n", mode.c_str(), cap1);
        if (mode == "zero") {
            CHECK(cap1 > 0 && cap1 < 1536);
        } else {
            CHECK(cap1 >= 1536);
        }
    }

    llama_model_free(model);
    llama_backend_free();

    if (n_fail == 0) {
        fprintf(stderr, "OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
