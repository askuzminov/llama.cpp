// fork: the rounding of the f16 hc read of qwen4exp (LLAMA_HC_F16) emulated on the CPU, against other formats and
// places of rounding, on a deep qwen4exp built from the 2-layer test model of test-llama-archs:
// - the model: n layers (48) in the layout of the real one, three gated delta net layers and then one QSA attention
//   layer, every tensor of the test model repeated by the kind of its layer, with fresh weights at the scale of an
//   initialization for training (N(0, 1/fan_in), the block output projections times --gain, norms ~1, scales 1). with
//   the N(0, 1) weights of the test model itself a deep model is chaotic: one f32 ulp anywhere changes everything
// - the rounding: the eval callback rounds a tensor in place right after its node is computed, through a ggml type
//   (round to nearest, as the GPUs convert), before anything reads it:
//     copy  the normalized hc streams before each sub-layer (hc_norm-<il>), what LLAMA_HC_F16 stores in f16 on
//           Vulkan, only in batches of 32 tokens and more (as the real path; decode stays f32)
//     res   the wide residual itself after each combine (hc_combine-<il>, l_last-<il>), every batch
//     ulp   the copy moved by one f32 ulp up or down instead: the control, the smallest change there is
// - the reference: the same model with the matmul weights rounded to f16 values and kept as F32, so the CPU multiplies
//   f32 activations; "vk" is that model with those weights as F16, where the CPU converts the activations to f16 for
//   each matmul, as the Vulkan mul_mm of a prompt keeps both operands in f16 (upstream on the 395)
// every variant reads the same random tokens: a prompt of -n positions in ubatches of --ub, then --decode single-token
// steps. per variant: KL(reference || variant) of the next-token distribution and the top-1 agreement by position
// bucket, and the relative L2 error of the residual after the layers (sampled rows) with its growth from layer 1.
//
// usage: test-hc-round-fork -m <dir>/qwen4exp-moe.gguf [-n 16384] [--decode 64] [--layers 48] [--gain 1]
//        [--topk 128] [--ub 512] [--seed 1] [-t 8] [--variants ulp,f16,bf16,f8,q8_0,f16res,bf16res,vk,vkf16]
// the test model: ctest's generate-models (build/tests/test-models) or test-llama-archs -a qwen4exp -o <dir>.
// 31.10, the Mac at the defaults (~20 min): KL at 8-16K ulp 6.1e-6, vk 1.50e-5, f16 1.66e-5, vkf16 1.66e-5, f16res
// 1.97e-5, bf16 2.74e-5, q8_0 5.58e-5, bf16res 6.88e-5, f8 2.92e-4; the residual error grows from layer 1 to 48
// 369x for f16 against 86824x for ulp (FORK.md, LLAMA_HC_F16)

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// the weights of a matmul: rounded to f16 values in both models, F16 in the "vk" one
static const char * const MATMUL_WEIGHTS[] = {
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_qkv.weight", "attn_gate.weight",
    "ssm_out.weight", "ssm_alpha.weight", "ssm_beta.weight", "ffn_gate_inp.weight", "ffn_down_exps.weight",
    "ffn_gate_up_exps.weight", "ffn_gate_shexp.weight", "ffn_down_shexp.weight", "ffn_up_shexp.weight",
    "indexer.q_proj.weight", "indexer.k_proj.weight", "hc_attn_down.weight", "hc_attn_up.weight",
    "hc_attn_inject.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
    "hc_attn_down_inject.weight", "hc_ffn_down_inject.weight", "output_hc_down.weight", "output_hc_up.weight",
    "output.weight",
};

// the projections that write a block's output into the residual
static const char * const OUT_PROJ[] = {
    "attn_output.weight", "ssm_out.weight", "ffn_down_exps.weight", "ffn_down_shexp.weight",
};

static bool ends_with(const std::string & s, const char * suffix) {
    const size_t n = strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

static bool is_matmul_weight(const std::string & name) {
    for (const char * m : MATMUL_WEIGHTS) {
        // the whole last two components, not a suffix of a longer name (hc_attn_down vs hc_attn_down_inject)
        if (ends_with(name, m) && (name.size() == strlen(m) || name[name.size() - strlen(m) - 1] == '.')) {
            return true;
        }
    }
    return false;
}

struct init_params {
    unsigned seed;
    float    gain;
    bool     f16_values; // the matmul weights rounded to f16 values (both models, so that they hold the same weights)
};

static const double PI = 3.14159265358979323846; // M_PI needs _USE_MATH_DEFINES on MSVC

// FNV-1a: std::hash differs between the standard libraries
static uint64_t name_hash(const std::string & s) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : s) {
        h = (h ^ c) * 0x100000001b3ull;
    }
    return h;
}

// the weights of one tensor, a function of its name and the seed: both models draw the same values, and every platform
// (std::normal_distribution is not the same algorithm everywhere, so Box-Muller over the fully specified mt19937_64)
static void set_tensor_data(ggml_tensor * t, void * userdata) {
    const init_params & p = *(const init_params *) userdata;
    const std::string name = t->name;
    const int64_t n = ggml_nelements(t);
    std::vector<float> v(n);

    std::mt19937_64 rng(name_hash(name) ^ ((uint64_t) p.seed * 0x9E3779B97F4A7C15ull));
    auto normal = [&](float mean, float stdev) {
        for (int64_t i = 0; i < n; i += 2) {
            const double u1 = ((rng() >> 11) + 1.0) * (1.0 / 9007199254740993.0); // (0, 1]
            const double u2 = (rng() >> 11) * (1.0 / 9007199254740992.0);         // [0, 1)
            const double r  = sqrt(-2.0 * log(u1));
            v[i] = (float) (mean + stdev * r * cos(2.0 * PI * u2));
            if (i + 1 < n) {
                v[i + 1] = (float) (mean + stdev * r * sin(2.0 * PI * u2));
            }
        }
    };

    const size_t dot = name.rfind('.');
    const size_t dot2 = dot == std::string::npos ? std::string::npos : name.rfind('.', dot - 1);
    const std::string comp = dot == std::string::npos ? name : name.substr(dot2 == std::string::npos ? 0 : dot2 + 1, dot - dot2 - 1);

    if (ends_with(name, ".scale") || ends_with(name, ".input_scale")) {
        std::fill(v.begin(), v.end(), 1.0f);
    } else if (ends_with(name, ".ssm_a")) {
        normal(0.0f, 1.0f);
        for (float & x : v) {
            x = -fabsf(x); // the state decays
        }
    } else if (name.find("norm") != std::string::npos && comp.find("norm") != std::string::npos) {
        normal(1.0f, 0.05f);
    } else if (ends_with(name, ".bias")) {
        normal(0.0f, 0.02f);
    } else if (name == "token_embd.weight" || name == "per_layer_token_embd.weight") {
        normal(0.0f, 1.0f);
    } else {
        float stdev = 1.0f / sqrtf((float) t->ne[0]);
        for (const char * o : OUT_PROJ) {
            if (ends_with(name, o)) {
                stdev *= p.gain;
            }
        }
        normal(0.0f, stdev);
    }

    const bool round16 = p.f16_values && is_matmul_weight(name);
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(n);
        for (int64_t i = 0; i < n; ++i) {
            h[i] = ggml_fp32_to_fp16(v[i]);
        }
        ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
    } else {
        GGML_ASSERT(t->type == GGML_TYPE_F32);
        if (round16) {
            for (float & x : v) {
                x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
            }
        }
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }
}

static size_t gguf_elem_size(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL:     return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:                        return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default:                                                            return 0;
    }
}

static void set_int_like(gguf_context * ctx, const gguf_context * src, const char * key, int64_t value) {
    const int64_t id = gguf_find_key(src, key);
    if (id >= 0 && gguf_get_kv_type(src, id) == GGUF_TYPE_INT32) {
        gguf_set_val_i32(ctx, key, (int32_t) value);
    } else {
        gguf_set_val_u32(ctx, key, (uint32_t) value);
    }
}

// the metadata of the deep model: the keys of the test model with n layers (an array per layer takes the value of the
// source layer of the same kind), the context and the indexer top-k given, and the tensors of the test model repeated
// by layer kind, F16 for the matmul weights when f16_matmul (the loader takes a tensor's type from here)
static gguf_context * deep_metadata(const gguf_context * src, ggml_context * src_ctx, int n_layer, int64_t n_ctx,
                                    int64_t topk, bool f16_matmul, ggml_context * meta_ctx) {
    const std::string arch = gguf_get_val_str(src, gguf_find_key(src, "general.architecture"));
    const std::string k_blocks = arch + ".block_count";
    const std::string k_recr   = arch + ".attention.recurrent_layers";
    const std::string k_ple    = arch + ".ple.layers";
    const int n_src = (int) gguf_get_val_u32(src, gguf_find_key(src, k_blocks.c_str()));

    // the kind of a layer: recurrent but every fourth (the real model: 36 gated delta net and 12 attention layers)
    const int64_t id_recr = gguf_find_key(src, k_recr.c_str());
    GGML_ASSERT(id_recr >= 0 && (int) gguf_get_arr_n(src, id_recr) == n_src);
    const gguf_type t_recr = gguf_get_arr_type(src, id_recr);
    auto src_recr = [&](int il) -> bool {
        const uint8_t * d = (const uint8_t *) gguf_get_arr_data(src, id_recr);
        const size_t es = gguf_elem_size(t_recr);
        int64_t x = 0;
        memcpy(&x, d + il*es, es);
        return x != 0;
    };
    int src_of[2] = { -1, -1 }; // [recurrent]: a source layer of that kind
    for (int il = 0; il < n_src; ++il) {
        if (src_of[src_recr(il)] < 0) {
            src_of[src_recr(il)] = il;
        }
    }
    GGML_ASSERT(src_of[0] >= 0 && src_of[1] >= 0 && "the test model needs a recurrent and an attention layer");
    std::vector<int> src_layer(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        src_layer[il] = src_of[il % 4 != 3 ? 1 : 0];
    }

    gguf_context * ctx = gguf_init_empty();
    gguf_set_kv(ctx, src);
    gguf_set_val_u32(ctx, k_blocks.c_str(), (uint32_t) n_layer);
    for (int64_t id = 0; id < gguf_get_n_kv(src); ++id) {
        const std::string key = gguf_get_key(src, id);
        if (gguf_get_kv_type(src, id) != GGUF_TYPE_ARRAY || key.rfind(arch + ".", 0) != 0 || key == k_ple ||
                (int) gguf_get_arr_n(src, id) != n_src) {
            continue;
        }
        const gguf_type at = gguf_get_arr_type(src, id);
        const size_t es = gguf_elem_size(at);
        if (es == 0) {
            continue;
        }
        const uint8_t * d = (const uint8_t *) gguf_get_arr_data(src, id);
        std::vector<uint8_t> out((size_t) n_layer * es);
        for (int il = 0; il < n_layer; ++il) {
            memcpy(out.data() + il*es, d + src_layer[il]*es, es);
        }
        gguf_set_arr_data(ctx, key.c_str(), at, out.data(), n_layer);
    }
    set_int_like(ctx, src, (arch + ".context_length").c_str(), n_ctx);
    set_int_like(ctx, src, (arch + ".attention.indexer.top_k").c_str(), topk);

    for (ggml_tensor * t = ggml_get_first_tensor(src_ctx); t; t = ggml_get_next_tensor(src_ctx, t)) {
        const std::string name = t->name;
        std::vector<std::string> names;
        if (name.rfind("blk.", 0) == 0) {
            const int il_src = atoi(name.c_str() + 4);
            const std::string rest = name.substr(name.find('.', 4) + 1);
            for (int il = 0; il < n_layer; ++il) {
                if (src_layer[il] == il_src && (il == 0 || rest.rfind("ple_", 0) != 0)) {
                    names.push_back("blk." + std::to_string(il) + "." + rest); // per-layer embeddings: layer 0 only
                }
            }
        } else {
            names.push_back(name);
        }
        for (const auto & nm : names) {
            const ggml_type type = f16_matmul && is_matmul_weight(nm) ? GGML_TYPE_F16 : GGML_TYPE_F32;
            ggml_tensor * m = ggml_new_tensor(meta_ctx, type, ggml_n_dims(t), t->ne);
            ggml_set_name(m, nm.c_str());
            gguf_add_tensor(ctx, m);
        }
    }
    return ctx;
}

// the rounding of one run, read by the eval callback
static ggml_type g_type    = GGML_TYPE_COUNT; // none
static bool      g_ulp     = false;
static bool      g_res     = false;
static int       g_sample  = 8;
static int64_t   g_rounded = 0;
static std::vector<std::vector<float>> g_rows; // per layer: the sampled rows of l_last, appended

static void round_through(ggml_type type, float * x, int64_t n, int64_t row) {
    const auto * tr = ggml_get_type_traits(type);
    std::vector<uint8_t> buf(ggml_row_size(type, row));
    for (int64_t i = 0; i < n; i += row) {
        tr->from_float_ref(x + i, buf.data(), row);
        tr->to_float(buf.data(), x + i, row);
    }
}

// the layer of a node named <prefix><il>, -1 for any other name (a view is "l_last-46 (reshaped)")
static int layer_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] == '\0') {
        return -1;
    }
    for (const char * c = name + n; *c; ++c) {
        if (*c < '0' || *c > '9') {
            return -1;
        }
    }
    return atoi(name + n);
}

static bool eval_cb(ggml_tensor * t, bool ask, void * /*user_data*/) {
    const int il_norm = layer_of(t->name, "hc_norm-");
    const int il_comb = layer_of(t->name, "hc_combine-");
    const int il_last = layer_of(t->name, "l_last-");
    const bool rounds = (g_type != GGML_TYPE_COUNT || g_ulp) &&
        ((!g_res && il_norm >= 0) || (g_res && (il_comb >= 0 || il_last >= 0)));
    if (ask) {
        return rounds || il_last >= 0;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->data != nullptr);
    float * x = (float *) t->data;
    const int64_t n = ggml_nelements(t);
    if (rounds) {
        const int64_t n_tok = il_norm >= 0 ? t->ne[1] : t->ne[2];
        if (g_res || n_tok >= 32) {
            if (g_ulp) {
                for (int64_t i = 0; i < n; ++i) {
                    const uint64_t h = (uint64_t) (i + 1) * 0x9E3779B97F4A7C15ull ^ (uint64_t) g_rounded * 0xBF58476D1CE4E5B9ull;
                    x[i] = nextafterf(x[i], (h >> 63) ? INFINITY : -INFINITY);
                }
            } else {
                round_through(g_type, x, n, t->ne[0]);
            }
            g_rounded++;
        }
    }
    if (il_last >= 0) {
        // [n_embd, hc, n_tok]: the last g_sample tokens of the batch
        const int64_t row   = t->ne[0] * t->ne[1];
        const int64_t n_tok = t->ne[2];
        const int64_t first = n_tok > g_sample ? n_tok - g_sample : 0;
        g_rows[il_last].insert(g_rows[il_last].end(), x + first*row, x + n_tok*row);
    }
    return true;
}

struct run_out {
    std::vector<float>              logits; // [positions, n_vocab]
    std::vector<std::vector<float>> rows;   // [layer][sampled rows * row]
};

static bool run(llama_model * model, const std::vector<llama_token> & tokens, int n, int ub, int n_threads, run_out & out) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = (uint32_t) tokens.size() + 256;
    cp.n_batch           = ub;
    cp.n_ubatch          = ub;
    cp.n_seq_max         = 1;
    cp.n_threads         = n_threads;
    cp.n_threads_batch   = n_threads;
    cp.no_perf           = true;
    cp.offload_kqv       = false;
    cp.op_offload        = false;
    cp.cb_eval           = eval_cb;
    cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create a context\n");
        return false;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    g_rows.assign(llama_model_n_layer(model), {});
    g_rounded = 0;
    out.logits.clear();
    out.logits.reserve(tokens.size() * n_vocab);

    llama_batch batch = llama_batch_init(ub, 0, 1);
    for (int pos = 0; pos < (int) tokens.size(); ) {
        const int nb = pos < n ? std::min(ub, n - pos) : 1;
        batch.n_tokens = nb;
        for (int i = 0; i < nb; ++i) {
            batch.token[i]     = tokens[pos + i];
            batch.pos[i]       = pos + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at %d\n", pos);
            llama_batch_free(batch);
            llama_free(ctx);
            return false;
        }
        const float * l = llama_get_logits(ctx);
        out.logits.insert(out.logits.end(), l, l + (size_t) nb * n_vocab);
        pos += nb;
    }
    llama_batch_free(batch);
    llama_free(ctx);
    out.rows = g_rows;
    return true;
}

static std::vector<double> log_softmax(const float * x, int n) {
    double m = x[0];
    for (int i = 1; i < n; ++i) {
        m = std::max(m, (double) x[i]);
    }
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += exp(x[i] - m);
    }
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] - m - log(s);
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string path, variants = "ulp,f16,bf16,f8,q8_0,f16res,bf16res,vk,vkf16";
    int n = 16384, n_decode = 64, n_layer = 48, ub = 512, n_threads = 8;
    int64_t topk = 128;
    float gain = 1.0f;
    unsigned seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "-m")         path = next();
        else if (a == "-n")         n = std::stoi(next());
        else if (a == "--decode")   n_decode = std::stoi(next());
        else if (a == "--layers")   n_layer = std::stoi(next());
        else if (a == "--gain")     gain = std::stof(next());
        else if (a == "--topk")     topk = std::stoll(next());
        else if (a == "--ub")       ub = std::stoi(next());
        else if (a == "--seed")     seed = (unsigned) std::stoul(next());
        else if (a == "-t")         n_threads = std::stoi(next());
        else if (a == "--variants") variants = next();
        else if (a == "--sample")   g_sample = std::stoi(next());
        else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 1;
        }
    }
    if (path.empty()) {
        fprintf(stderr, "usage: %s -m <dir>/qwen4exp-moe.gguf [-n 16384] [--decode 64] [--layers 48] [--gain 1] "
                        "[--topk 128] [--ub 512] [--seed 1] [-t 8] [--variants ...]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    ggml_context * src_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &src_ctx };
    gguf_context * src = gguf_init_from_file(path.c_str(), gp);
    if (!src) {
        fprintf(stderr, "cannot read %s\n", path.c_str());
        return 1;
    }
    ggml_init_params mp_ctx = { ggml_tensor_overhead() * 65536, nullptr, true };
    ggml_context * meta_ctx = ggml_init(mp_ctx);

    const int64_t n_ctx = n + n_decode + 256;
    init_params ip = { seed, gain, true };
    // everything on the CPU, also in a GPU build: the callback reads and writes the tensors in host memory
    static ggml_backend_dev_t no_devices[] = { nullptr };
    llama_model_params mp = llama_model_default_params();
    mp.devices      = no_devices;
    mp.n_gpu_layers = 0;

    gguf_context * meta_a = deep_metadata(src, src_ctx, n_layer, n_ctx, topk, false, meta_ctx);
    gguf_context * meta_c = deep_metadata(src, src_ctx, n_layer, n_ctx, topk, true,  meta_ctx);
    llama_model * model_a = llama_model_init_from_user(meta_a, set_tensor_data, &ip, mp);
    llama_model * model_c = llama_model_init_from_user(meta_c, set_tensor_data, &ip, mp);
    if (!model_a || !model_c) {
        fprintf(stderr, "cannot build the deep model\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_a));

    std::mt19937 rng(seed);
    std::vector<llama_token> tokens(n + n_decode);
    for (auto & t : tokens) {
        t = (llama_token) (rng() % (unsigned) n_vocab);
    }

    printf("deep qwen4exp: %d layers (every fourth QSA attention), gain %.2f, top-k %lld, %d + %d positions in ubatches "
           "of %d, CPU\n", n_layer, gain, (long long) topk, n, n_decode, ub);

    run_out ref;
    if (!run(model_a, tokens, n, ub, n_threads, ref)) {
        return 1;
    }

    // the position buckets: doubling from 1K to the prompt end, then the decode steps
    std::vector<std::pair<int, int>> buckets;
    std::vector<std::string> bucket_names;
    for (int a = 0, b = 1024; a < n; a = b, b *= 2) {
        buckets.emplace_back(a, std::min(b, n));
        bucket_names.push_back(std::to_string(a / 1024) + "-" + std::to_string(std::min(b, n) / 1024) + "K");
    }
    if (n_decode > 0) {
        buckets.emplace_back(n, n + n_decode);
        bucket_names.push_back("decode");
    }
    const int layers_shown[] = { 0, n_layer / 8, n_layer / 4, n_layer / 2, 3 * n_layer / 4, n_layer - 1 };

    std::vector<std::vector<double>> ref_ls(tokens.size());
    for (size_t p = 0; p < tokens.size(); ++p) {
        ref_ls[p] = log_softmax(ref.logits.data() + p*n_vocab, n_vocab);
    }

    printf("\nKL(reference || variant) / top-1 agreement %% by position\n%-8s", "variant");
    for (const auto & b : bucket_names) {
        printf(" %18s", b.c_str());
    }
    printf("   %-31s", "residual rel. error after layer");
    for (int il : layers_shown) {
        printf(" %7d", il + 1);
    }
    printf("   growth\n");

    size_t start = 0;
    while (start < variants.size()) {
        size_t end = variants.find(',', start);
        if (end == std::string::npos) {
            end = variants.size();
        }
        const std::string v = variants.substr(start, end - start);
        start = end + 1;

        llama_model * model = model_a;
        g_type = GGML_TYPE_COUNT;
        g_ulp  = false;
        g_res  = false;
        std::string fmt = v;
        if (fmt.rfind("vk", 0) == 0) {
            model = model_c;
            fmt = fmt.substr(2);
        }
        if (ends_with(fmt, "res")) {
            g_res = true;
            fmt = fmt.substr(0, fmt.size() - 3);
        }
        if      (fmt == "f16")  g_type = GGML_TYPE_F16;
        else if (fmt == "bf16") g_type = GGML_TYPE_BF16;
        else if (fmt == "f8")   g_type = GGML_TYPE_F8_E4M3;
        else if (fmt == "q8_0") g_type = GGML_TYPE_Q8_0;
        else if (fmt == "ulp")  g_ulp  = true;
        else if (!fmt.empty()) {
            fprintf(stderr, "unknown variant %s\n", v.c_str());
            return 1;
        }

        run_out out;
        if (!run(model, tokens, n, ub, n_threads, out)) {
            return 1;
        }

        printf("%-8s", v.c_str());
        for (const auto & [a, b] : buckets) {
            double kl = 0.0;
            int top = 0;
            for (int p = a; p < b; ++p) {
                const std::vector<double> ls = log_softmax(out.logits.data() + (size_t) p*n_vocab, n_vocab);
                const std::vector<double> & lr = ref_ls[p];
                int am_r = 0, am_v = 0;
                for (int i = 0; i < n_vocab; ++i) {
                    kl += exp(lr[i]) * (lr[i] - ls[i]);
                    am_r = lr[i] > lr[am_r] ? i : am_r;
                    am_v = ls[i] > ls[am_v] ? i : am_v;
                }
                top += am_r == am_v;
            }
            printf(" %9.2e %7.2f%%", kl / (b - a), 100.0 * top / (b - a));
        }

        // the residual: prompt rows only (the last g_sample positions of each ubatch), relative L2 error per row
        const int64_t row = (int64_t) (ref.rows[0].size() / (((n + ub - 1) / ub) * (size_t) g_sample + n_decode));
        const int64_t n_prompt_rows = ((n + ub - 1) / ub) * (int64_t) g_sample;
        std::vector<double> err(n_layer, 0.0);
        for (int il = 0; il < n_layer; ++il) {
            for (int64_t r = 0; r < n_prompt_rows; ++r) {
                double d2 = 0.0, r2 = 0.0;
                for (int64_t i = 0; i < row; ++i) {
                    const double rv = ref.rows[il][r*row + i];
                    const double dv = out.rows[il][r*row + i] - rv;
                    d2 += dv*dv;
                    r2 += rv*rv;
                }
                err[il] += sqrt(d2 / std::max(r2, 1e-30));
            }
            err[il] /= n_prompt_rows;
        }
        printf("   %31s", "");
        for (int il : layers_shown) {
            printf(" %7.1e", err[il]);
        }
        printf("   %6.0fx  (%lld rounded)\n", err[n_layer - 1] / std::max(err[0], 1e-30), (long long) g_rounded);
        fflush(stdout);
    }

    llama_model_free(model_a);
    llama_model_free(model_c);
    gguf_free(meta_a);
    gguf_free(meta_c);
    gguf_free(src);
    ggml_free(meta_ctx);
    ggml_free(src_ctx);
    llama_backend_free();
    return 0;
}
