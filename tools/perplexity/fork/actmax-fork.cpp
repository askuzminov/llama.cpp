// fork: the largest activation that each kind of matmul reads (src1 of MUL_MAT and MUL_MAT_ID), and how many values
// lie outside the f16 range. The Vulkan f16reg paths read their activations as f16 (an f16 copy or a conversion in the
// tile, ggml-vulkan-ops-fork.cpp), where |x| > 65504 would be inf. Runs the text of -f through the model:
//
//   llama-actmax-fork -m model.gguf -f text.txt -c 4096 [-ngl 99] [--actmax-tokens N]
//
// prints one line per kind of node (the name without its layer), the largest |x| over all layers and tokens with its
// layer, the values above 65504 and above 32768, and the values read. Each matmul is a sync point of the scheduler, so a
// run is slower than a perplexity run of the same text.

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct actmax_stat {
    double      max_abs = 0.0;
    std::string max_node;
    int64_t     over_f16  = 0; // |x| > 65504: inf as f16
    int64_t     over_half = 0; // |x| > 32768
    int64_t     n_values  = 0;
    int64_t     n_calls   = 0;
};

struct actmax_state {
    std::map<std::string, actmax_stat> stats;
    std::vector<uint8_t>               row;
};

// "ffn_moe_down-12" -> "ffn_moe_down"
std::string actmax_kind(const ggml_tensor * t) {
    std::string name = t->name;
    const size_t dash = name.rfind('-');
    if (dash != std::string::npos && dash + 1 < name.size() &&
            std::all_of(name.begin() + dash + 1, name.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        name.resize(dash);
    }
    return std::string(ggml_op_name(t->op)) + " " + name;
}

void actmax_row(actmax_stat & s, const ggml_tensor * x, const uint8_t * p, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const double v = x->type == GGML_TYPE_F32 ? (double) ((const float *) p)[i]
                                                  : (double) ggml_fp16_to_fp32(((const ggml_fp16_t *) p)[i]);
        const double a = std::fabs(v);
        if (!(a <= s.max_abs)) {
            s.max_abs = a; // NaN counts as the largest, it shows as nan
        }
        s.over_f16  += a > 65504.0;
        s.over_half += a > 32768.0;
    }
    s.n_values += n;
}

bool actmax_cb(ggml_tensor * t, bool ask, void * user_data) {
    const bool mm = t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID;
    if (ask) {
        return mm;
    }
    if (!mm) {
        return true;
    }
    const ggml_tensor * x = t->src[1];
    if (x == nullptr || (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16) || x->buffer == nullptr ||
            x->nb[0] != ggml_type_size(x->type)) {
        return true;
    }
    auto * st = (actmax_state *) user_data;
    actmax_stat & s = st->stats[actmax_kind(t)];
    const double before = s.max_abs;
    const size_t row_bytes = x->ne[0] * ggml_type_size(x->type);
    const bool host = ggml_backend_buffer_is_host(x->buffer);
    st->row.resize(row_bytes);
    for (int64_t i3 = 0; i3 < x->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < x->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < x->ne[1]; ++i1) {
                const size_t off = i1*x->nb[1] + i2*x->nb[2] + i3*x->nb[3];
                const uint8_t * p = nullptr;
                if (host) {
                    p = (const uint8_t *) x->data + off;
                } else {
                    ggml_backend_tensor_get(x, st->row.data(), off, row_bytes);
                    p = st->row.data();
                }
                actmax_row(s, x, p, x->ne[0]);
            }
        }
    }
    if (!(s.max_abs <= before)) {
        s.max_node = t->name;
    }
    s.n_calls++;
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    common_params params;
    int64_t n_tokens_max = -1;
    // --actmax-tokens N: how many tokens of the text to run (default: one context)
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--actmax-tokens") == 0 && i + 1 < argc) {
            n_tokens_max = std::atoll(argv[++i]);
            continue;
        }
        args.push_back(argv[i]);
    }
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    if (params.prompt.empty()) {
        fprintf(stderr, "%s: give the text with -f\n", __func__);
        return 1;
    }

    actmax_state state;
    params.cb_eval           = actmax_cb;
    params.cb_eval_user_data = &state;
    params.warmup            = false;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s: failed to load the model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    const int64_t n_ctx   = llama_n_ctx(ctx);
    const int64_t n_batch = std::min<int64_t>(llama_n_batch(ctx), n_ctx);
    int64_t n_run = std::min<int64_t>((int64_t) tokens.size(), n_tokens_max > 0 ? n_tokens_max : n_ctx);
    n_run = std::min(n_run, n_ctx);
    (void) vocab;

    fprintf(stderr, "%s: %lld tokens in batches of %lld\n", __func__, (long long) n_run, (long long) n_batch);
    for (int64_t i = 0; i < n_run; i += n_batch) {
        const int32_t n = (int32_t) std::min(n_batch, n_run - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n)) != 0) {
            fprintf(stderr, "%s: llama_decode failed at token %lld\n", __func__, (long long) i);
            return 1;
        }
    }

    std::vector<std::pair<std::string, actmax_stat>> rows(state.stats.begin(), state.stats.end());
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
        return !(a.second.max_abs <= b.second.max_abs); // NaN first
    });
    printf("%-44s %12s %-28s %10s %10s %14s\n", "node (op, name without the layer)", "max |x|", "at", ">65504", ">32768", "values");
    for (const auto & [kind, s] : rows) {
        printf("%-44s %12.1f %-28s %10lld %10lld %14lld\n", kind.c_str(), s.max_abs, s.max_node.c_str(),
               (long long) s.over_f16, (long long) s.over_half, (long long) s.n_values);
    }
    return 0;
}
