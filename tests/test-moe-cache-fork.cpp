// the device cache of the host experts with --phase-mem (llama_moe_cache_fork, src/fork/llama-moecache-fork.cpp,
// src/fork/llama-context-fork.cpp), on the first GPU, against the behaviour before 25.10:
// - keep: a prompt phase takes whole layers of slots for its buffers and leaves the rest, the next generation phase
//   uploads only the layers it took, and the prompt copies take the experts the slots hold from device memory. the
//   same experts on the same devices: the logits of the generated tokens bit for bit
// - fill: the layers it took are filled in the background during the next steps, the graphs compute the experts on
//   the host until then: the logits within the difference of the host and the device computing an expert
// - kv: a growth of the elastic KV takes layers of slots one by one where the device is full (LLAMA_KV_ELASTIC_TEST_MIB
//   makes it full), the next step gives them back
// - upload: as keep, and the uploads ahead of whole weights (GGML_SCHED_PREFETCH, every prompt here with
//   GGML_SCHED_PREFETCH_LOAD=0) take the experts the slots hold from device memory too (LLAMA_MOE_CACHE_COPY=2)
// the reference run (--out) releases all slots at every phase switch and uploads every expert of a prompt copy from the
// host. the knobs are read once per process, so each mode is a run of its own (ctest runs them after the reference).
// the model is the qwen4exp of test-llama-archs with 32 experts instead of 2 and without the per-expert scales, which
// the cache does not take; its memory takes the elastic KV. each run checks in the log that its path ran
//
// usage: test-moe-cache-fork -m <dir>/qwen4exp-moe.gguf (--out <file> | --ref <file> --mode keep|fill|kv|upload)

#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const char * what, int line) {
    if (!ok) {
        fprintf(stderr, "line %d: %s\n", line, what);
        n_fail++;
    }
}

#define CHECK(x) check((x), #x, __LINE__)

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static std::mutex  g_log_mutex;
static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void * /*user_data*/) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log += text;
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static bool log_has(const char * what) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    return g_log.find(what) != std::string::npos;
}

static const int N_EXPERT = 32;
static const int N_SLOTS  = 32; // per layer: 33 slots of 1.125 MiB on each of the 2 layers, 74.25 MiB

// the qwen4exp of test-llama-archs with N_EXPERT experts: a random router, so that the routing spreads, the two
// experts tiled with their rows shuffled, no per-expert scales, a context of 4096
static bool make_model(const char * src, const std::string & dst) {
    ggml_context * meta = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &meta };
    gguf_context * in = gguf_init_from_file(src, params);
    if (in == nullptr) {
        fprintf(stderr, "cannot read %s\n", src);
        return false;
    }
    const std::string arch = gguf_get_val_str(in, gguf_find_key(in, "general.architecture"));

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    gguf_set_val_u32(out, (arch + ".expert_count").c_str(), N_EXPERT);
    gguf_set_val_u32(out, (arch + ".context_length").c_str(), 4096);
    if (gguf_find_key(in, (arch + ".rope.scaling.original_context_length").c_str()) >= 0) {
        gguf_set_val_u32(out, (arch + ".rope.scaling.original_context_length").c_str(), 4096);
    }

    // the bytes of the new tensors: the experts and the router for N_EXPERT, the rest as they are
    const int64_t n_tensors = gguf_get_n_tensors(in);
    size_t size = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const ggml_tensor * t = ggml_get_tensor(meta, gguf_get_tensor_name(in, i));
        const size_t n = t->ne[2] > 1 ? ggml_nbytes(t)/t->ne[2]*N_EXPERT : t->ne[1] > 1 ? ggml_nbytes(t)/t->ne[1]*N_EXPERT : 0;
        size += std::max(ggml_nbytes(t), n) + GGML_MEM_ALIGN;
    }
    ggml_init_params ip = {
        /*.mem_size   =*/ (size_t) (n_tensors + 1)*ggml_tensor_overhead() + size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(ip);

    std::mt19937 rng(42);
    std::normal_distribution<float> router(0.0f, 0.5f);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(in, i);
        const ggml_tensor * t = ggml_get_tensor(meta, name);
        const std::string n(name);
        if (n.find("exps.scale") != std::string::npos || n.find("exps.input_scale") != std::string::npos) {
            continue;
        }
        ggml_tensor * u = nullptr;
        if (n.size() > 19 && n.compare(n.size() - 19, 19, "ffn_gate_inp.weight") == 0) {
            GGML_ASSERT(t->type == GGML_TYPE_F32);
            u = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t->ne[0], N_EXPERT);
            float * d = (float *) u->data;
            for (int64_t k = 0; k < ggml_nelements(u); ++k) {
                d[k] = router(rng);
            }
        } else if (n.find("_exps.weight") != std::string::npos) {
            u = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], N_EXPERT);
            std::vector<int64_t> rows(t->ne[1]);
            for (int64_t e = 0; e < N_EXPERT; ++e) {
                for (int64_t r = 0; r < t->ne[1]; ++r) {
                    rows[r] = r;
                }
                std::shuffle(rows.begin(), rows.end(), rng);
                const char * from = (const char *) t->data + (e % t->ne[2])*t->nb[2];
                char       * to   = (char *) u->data + e*u->nb[2];
                for (int64_t r = 0; r < t->ne[1]; ++r) {
                    memcpy(to + r*u->nb[1], from + rows[r]*t->nb[1], t->nb[1]);
                }
            }
        } else {
            u = ggml_dup_tensor(ctx, t);
            memcpy(u->data, t->data, ggml_nbytes(t));
        }
        ggml_set_name(u, name);
        gguf_add_tensor(out, u);
    }

    const bool ok = gguf_write_to_file(out, dst.c_str(), false);
    if (!ok) {
        fprintf(stderr, "cannot write %s\n", dst.c_str());
    }
    gguf_free(out);
    gguf_free(in);
    ggml_free(ctx);
    ggml_free(meta);
    return ok;
}

struct record {
    int32_t token;
    int32_t id[8];
    float   logprob[8];
};

// six turns of 180 new tokens and 32 greedy ones, as a server conversation; the top 8 of every generated token
static bool run(const std::string & path, ggml_backend_dev_t gpu, std::vector<record> & recs) {
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { gpu, nullptr };
    const llama_model_tensor_buft_override overrides[] = {
        { "exps", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    mp.n_gpu_layers          = 99;
    mp.devices               = devs;
    mp.tensor_buft_overrides = overrides;
    mp.load_mode             = LLAMA_LOAD_MODE_NONE; // the experts in a host buffer of their own, not a mapping

    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", path.c_str());
        return false;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx       = 2048;
    cp.n_batch     = 256;
    cp.n_ubatch    = 256;
    cp.n_seq_max   = 1;
    cp.phase_mem   = true;
    cp.n_moe_cache = N_SLOTS;
    cp.no_perf     = true;

    llama_context * lctx = llama_init_from_model(model, cp);
    if (lctx == nullptr) {
        fprintf(stderr, "cannot create the context\n");
        llama_model_free(model);
        return false;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> pick(3, n_vocab - 1);

    llama_batch batch = llama_batch_init(256, 0, 1);
    int pos = 0;
    bool ok = true;

    auto decode = [&](const std::vector<llama_token> & toks) {
        batch.n_tokens = 0;
        for (size_t i = 0; i < toks.size(); ++i) {
            batch.token   [batch.n_tokens] = toks[i];
            batch.pos     [batch.n_tokens] = pos++;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id  [batch.n_tokens][0] = 0;
            batch.logits  [batch.n_tokens] = i + 1 == toks.size();
            batch.n_tokens++;
        }
        return llama_decode(lctx, batch) == 0;
    };

    for (int turn = 0; turn < 6 && ok; ++turn) {
        std::vector<llama_token> prompt(180);
        for (auto & t : prompt) {
            t = pick(rng);
        }
        ok = decode(prompt);
        for (int g = 0; g < 32 && ok; ++g) {
            const float * logits = llama_get_logits_ith(lctx, -1);
            const float   mx     = *std::max_element(logits, logits + n_vocab);
            double sum = 0.0;
            for (int k = 0; k < n_vocab; ++k) {
                sum += std::exp((double) logits[k] - mx);
            }
            const float lse = mx + (float) std::log(sum);

            std::vector<int32_t> ids(n_vocab);
            for (int k = 0; k < n_vocab; ++k) {
                ids[k] = k;
            }
            std::partial_sort(ids.begin(), ids.begin() + 8, ids.end(), [&](int32_t a, int32_t b) {
                return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
            });

            record r = {};
            r.token = ids[0];
            for (int k = 0; k < 8; ++k) {
                r.id[k]      = ids[k];
                r.logprob[k] = logits[ids[k]] - lse;
            }
            recs.push_back(r);

            ok = decode({ (llama_token) r.token });
        }
    }

    llama_batch_free(batch);
    llama_free(lctx); // the cache prints its report on the way out
    llama_model_free(model);

    if (!ok) {
        fprintf(stderr, "decode failed\n");
    }
    return ok;
}

int main(int argc, char ** argv) {
    const char * model_in = nullptr;
    const char * out      = nullptr;
    const char * ref      = nullptr;
    std::string  mode     = "ref";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_in = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out = argv[++i];
        } else if (strcmp(argv[i], "--ref") == 0 && i + 1 < argc) {
            ref = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        }
    }
    if (model_in == nullptr || (out == nullptr) == (ref == nullptr) ||
            (ref != nullptr && mode != "keep" && mode != "fill" && mode != "kv" && mode != "upload")) {
        fprintf(stderr, "usage: %s -m <qwen4exp-moe.gguf> (--out <file> | --ref <file> --mode keep|fill|kv|upload)\n", argv[0]);
        return 1;
    }

    // the prompts of 180 tokens take the prompt phase, the LRU makes the experts of a release and those of a layer that
    // kept its slots the same, and the elastic KV grows every 256 cells
    set_env("LLAMA_PHASE_SMALL_PROMPT", "0");
    set_env("LLAMA_MOE_CACHE_POLICY", "lru");
    set_env("LLAMA_KV_ELASTIC", "1");
    set_env("LLAMA_KV_ELASTIC_STEP", "256");
    if (out != nullptr) {
        set_env("LLAMA_MOE_CACHE_KEEP", "0");
        set_env("LLAMA_MOE_CACHE_COPY", "0");
    } else if (mode == "keep" || mode == "fill" || mode == "upload") {
        // one of the two layers of 37.1 MiB stays (the last one goes first: the counts are even), so the 1-token MoE of
        // the last layer in a prompt graph (its output row) has no slots, as in the reference
        set_env("LLAMA_MOE_CACHE_KEEP", "40");
        set_env("LLAMA_MOE_CACHE_FILL", mode == "fill" ? "1" : "0");
        if (mode == "upload") {
            set_env("LLAMA_MOE_CACHE_COPY", "2");
            set_env("GGML_SCHED_PREFETCH_LOAD", "0");
        }
    } else {
        // the slots take 74.25 MiB, so the KV gets 512 cells next to them and every growth past that takes a layer
        set_env("LLAMA_KV_ELASTIC_TEST_MIB", "75");
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && gpu == nullptr; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(d);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu = d;
        }
    }

    const std::string base     = out != nullptr ? out : ref;
    const std::string model    = base + ".gguf";
    const std::string recs_out = base;

    if (gpu == nullptr) {
        // the reference leaves an empty file, so the modes after it skip too
        if (out != nullptr) {
            FILE * f = fopen(recs_out.c_str(), "wb");
            if (f != nullptr) {
                fclose(f);
            }
        }
        printf("no GPU device, the moe-cache needs one: skipped\n");
        llama_backend_free();
        return 0;
    }

    if (out != nullptr && !make_model(model_in, model)) {
        llama_backend_free();
        return 1;
    }

    std::vector<record> recs;
    if (!run(model, gpu, recs)) {
        llama_backend_free();
        return 1;
    }

    if (out != nullptr) {
        FILE * f = fopen(recs_out.c_str(), "wb");
        CHECK(f != nullptr);
        if (f != nullptr) {
            CHECK(fwrite(recs.data(), sizeof(record), recs.size(), f) == recs.size());
            fclose(f);
        }
        CHECK(log_has("moe_cache: ") && log_has("slots per layer"));
        printf("%s: %zu tokens, the reference%s\n", argv[0], recs.size(), n_fail == 0 ? "" : ", FAILED");
        llama_backend_free();
        return n_fail == 0 ? 0 : 1;
    }

    std::vector<record> want;
    {
        FILE * f = fopen(recs_out.c_str(), "rb");
        CHECK(f != nullptr);
        if (f != nullptr) {
            record r;
            while (fread(&r, sizeof(r), 1, f) == 1) {
                want.push_back(r);
            }
            fclose(f);
        }
    }
    if (want.empty()) {
        printf("the reference ran without a GPU: skipped\n");
        llama_backend_free();
        return 0;
    }

    // keep is bit for bit; fill computes the experts that wait on the host (3.5e-5 apart on the Mac Vulkan0); kv keeps
    // all slots through the prompts, so the 1-token MoE of the last layer reads them there (4.8e-7)
    const double tol = mode == "keep" || mode == "upload" ? 0.0 : mode == "fill" ? 1e-4 : 1e-5;

    CHECK(recs.size() == want.size());
    double max_diff = 0.0;
    for (size_t i = 0; i < std::min(recs.size(), want.size()); ++i) {
        if (recs[i].token != want[i].token) {
            fprintf(stderr, "token %zu: %d against %d of the reference\n", i, recs[i].token, want[i].token);
            n_fail++;
            break;
        }
        for (int a = 0; a < 8; ++a) {
            for (int b = 0; b < 8; ++b) {
                if (recs[i].id[a] == want[i].id[b]) {
                    max_diff = std::max(max_diff, (double) std::fabs(recs[i].logprob[a] - want[i].logprob[b]));
                }
            }
        }
    }
    if (max_diff > tol) {
        fprintf(stderr, "logprobs %.3g apart, more than %.3g\n", max_diff, tol);
        n_fail++;
    }

    // the path of the mode ran
    CHECK(log_has("kept their slots")); // an alloc() that uploaded only the layers it had taken
    if (mode == "keep" || mode == "fill" || mode == "upload") {
        CHECK(log_has("(moe-cache kept 37 of 74 MiB)"));
        CHECK(log_has("prompt copies: "));
    }
    if (mode == "upload") {
        CHECK(!log_has("the uploads ahead 0.0 MiB")); // upload_weight() took whole weights
    }
    if (mode == "fill") {
        CHECK(log_has("filled in the background"));
        CHECK(log_has("the background fill uploaded"));
    }
    if (mode == "kv") {
        CHECK(log_has("kv_elastic: 512 -> 768 cells"));
    }

    printf("%s %s: %zu tokens, logprobs %.3g apart%s\n", argv[0], mode.c_str(), recs.size(), max_diff, n_fail == 0 ? "" : ", FAILED");
    llama_backend_free();
    return n_fail == 0 ? 0 : 1;
}
