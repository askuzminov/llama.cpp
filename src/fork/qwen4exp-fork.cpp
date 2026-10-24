#include "qwen4exp-fork.h"
#include "llama-fork.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx-fork.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>

// [TAG_QWEN4_REIMPLEMENT]
// TODO: this graph implementation is pending complete reimplementation - do not use it as a reference

// the PLE rows gathered ahead, see ple_prefetch_start
struct llama_model_qwen4exp_fork::ple_prefetch {
    std::mutex           mutex;
    std::thread          worker;
    bool                 active = false;
    std::vector<int32_t> idx;
    std::vector<float>   rows;

    int64_t n_started = 0;
    int64_t n_used    = 0;

    void finish() {
        if (worker.joinable()) {
            worker.join();
        }
    }

    ~ple_prefetch() { finish(); }
};

struct llama_model_qwen4exp_fork::ple_warmer {
    std::mutex                        mutex;
    std::condition_variable           cv;
    std::thread                       worker;
    std::vector<std::vector<int32_t>> queue;
    bool                              stop = false;

    int64_t n_rows = 0;

    ~ple_warmer() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
    }
};

llama_model_qwen4exp_fork::llama_model_qwen4exp_fork(const struct llama_model_params & params) : llama_model_base(params) {}

int64_t llama_model_qwen4exp_fork::mtp_window() {
    static const int64_t value = [] {
        const char * env = std::getenv("LLAMA_MTP_WINDOW");
        return env ? (int64_t) std::atoll(env) : (int64_t) 16384;
    }();
    return value;
}

bool llama_model_qwen4exp_fork::mtp_window_kv(const llama_cparams & cparams, uint32_t & kv_size, uint32_t & n_swa, llama_swa_type & swa_type) {
    static const int mode = [] {
        const char * env = std::getenv("LLAMA_MTP_WINDOW_KV");
        return env ? std::atoi(env) : -1;
    }();
    const int64_t window = mtp_window();
    if (window <= 0 || mode == 0 || (mode < 0 && cparams.n_seq_max <= 1)) {
        return false;
    }
    // as llama_kv_cache_iswa sizes its SWA cache: the window of every sequence plus a ubatch
    const int64_t size = window*(cparams.kv_unified ? cparams.n_seq_max : 1) + cparams.n_ubatch;
    if (size >= (int64_t) cparams.n_ctx_seq) {
        return false;
    }
    kv_size  = (uint32_t) size;
    n_swa    = (uint32_t) window;
    swa_type = LLAMA_SWA_TYPE_STANDARD;
    return true;
}

llama_model_qwen4exp_fork::~llama_model_qwen4exp_fork() {
    if (ple_pf && ple_pf->n_started > 0) {
        LLAMA_LOG_INFO("%s: PLE rows of %" PRId64 " ubatches gathered ahead, %" PRId64 " of them used\n", __func__,
                ple_pf->n_started, ple_pf->n_used);
    }
    if (ple_wm && ple_wm->n_rows > 0) {
        LLAMA_LOG_INFO("%s: %" PRId64 " PLE rows of coming tokens read ahead\n", __func__, ple_wm->n_rows);
    }
    if (ple_cache) {
        ple_cache->print_stats("PLE table");
    }
}

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

// a shared draft head keeps no embedding table and no LM head of its own: it reads them out of the
// model it drafts for, reached through the target context
static const llama_model & qwen4exp_target_model(const llama_cparams & cparams, const llama_model & model, const char * name) {
    if (cparams.ctx_other == nullptr) {
        throw std::runtime_error(format("this qwen4exp draft head has no '%s' of its own; "
                                        "load it with -md against the model it drafts for", name));
    }

    const llama_model & other = *llama_get_model(cparams.ctx_other);
    if (other.hparams.n_embd != model.hparams.n_embd || other.vocab.n_tokens() != model.vocab.n_tokens()) {
        throw std::runtime_error(format("the qwen4exp draft head and the target model disagree on the shape of '%s'", name));
    }

    return other;
}

void llama_model_qwen4exp_fork::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // the MTP block attends densely by default (graph_mtp), whatever ratio the file gives it: our converter writes 0,
    // the upstream one the trunk ratio. LLAMA_MTP_QSA=1: it takes the trunk ratio and attends through QSA like a
    // full-attention trunk layer, its draft context then holds a llama_memory_hybrid_idx_fork (llama_model::create_memory)
    {
        const char * env = std::getenv("LLAMA_MTP_QSA");
        const bool mtp_qsa = env != nullptr && std::atoi(env) != 0;

        uint32_t r_trunk = 0;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            r_trunk = std::max(r_trunk, hparams.dsv4_compress_ratios[il]);
        }
        for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
            hparams.dsv4_compress_ratios[il] = mtp_qsa ? r_trunk : 0;
        }
    }

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp_fork::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    // a draft-only sidecar (converted with --mtp) carries the head, the embedding table and the
    // LM head, but none of the trunk; a trunk-only file declares the head without its tensors
    // (upstream 448147d42: load_mtp is set for the main model whenever draft-mtp is asked for, -md or not)
    const auto nf           = nextn_flags(ml, LLM_TENSOR_HC_ATTN_NORM);
    const bool mtp_only     = (nf.trunk & TENSOR_NOT_REQUIRED) != 0;
    const int  trunk_flags  = nf.trunk;
    // a hand-made shared head carries neither, so the graph reads both out of the target
    const bool borrows      = mtp_only && ml.get_weight("token_embd.weight") == nullptr;
    const int  borrow_flags = borrows ? TENSOR_NOT_REQUIRED | TENSOR_BORROWED : trunk_flags;
    // MTP tensors sit in the trailing blocks: skipped unless a draft head was asked for, optional in a trunk-only file
    const int  mtp_flags    = nf.mtp;

    // a shared draft head has no table of its own, the graph borrows the target one
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, borrow_flags);

    // there is no output_norm: the final hyper-connection mixer carries it
    // the gammas load as [n_embd, hc] so the grouped norm multiplies them without a graph reshape
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | trunk_flags);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | borrow_flags);
    // never tie to a token_embd that a borrowing draft head does not have
    if (output == NULL && tok_embd != NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows },
                                           TENSOR_READ_LAZY | TENSOR_READ_CACHE);

        // -lzm dio creates no tensor, the rows come out of the file through the row cache
        ple_cached = per_layer_tok_embd == nullptr;

        if (ple_cached && !ml.no_alloc) {
            const auto * loc = ml.lazy.get_location(ple_name);
            GGML_ASSERT(loc != nullptr && loc->idx < ml.file_paths.size());

            ple_cache.reset(new llama_row_cache(ml.file_paths[loc->idx], loc->offs, loc->type,
                                                loc->ne0, loc->ne1, /*n_threads =*/ 0));
        }
    }

    for (int il = 0; il < n_layer_all; ++il) {
        auto & layer = layers[il];

        // the MTP block is structurally a trunk block: is_recr()/is_ple() are both false past
        // the trunk, so it takes the full-attention + MoE path below with no special casing
        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (il < n_layer) {
            continue;
        }

        // NextN/MTP head. enorm/hnorm gate the two inputs; eh_proj is the checkpoint's
        // fc_embedding and fc_hidden fused side by side, so one matmul over concat(e, h)
        // computes fc_embedding@e + fc_hidden@h.
        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);

        // the head's own output mixer, mirroring the trunk's hc_head_*: it collapses the hc
        // streams and stands in for the output norm, of which qwen4exp has none
        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags);
        layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp_fork::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// DFlash drafters read the layer inputs as [n_embd, T] rows, so collapse the streams
// (upstream 18a0777a5 on gg/qwen4exp-dflash-layer-inp, not in master yet)
static ggml_tensor * hc_stream_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t n_embd = x->ne[0];
    const int64_t hc     = x->ne[1];
    const int64_t nt     = x->ne[2];

    ggml_tensor * acc = ggml_view_2d(ctx, x, n_embd, nt, x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, n_embd, nt, x->nb[2], s*x->nb[1]));
    }

    return ggml_scale(ctx, acc, 1.0f / (float) hc);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp_fork::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);

    // the normalized streams in f16 (LLAMA_HC_F16, 0 = f32 as upstream). three nodes read them (the down projection, the
    // gated mean and the injection), each a pass over hc_dim floats per token; the reference keeps the whole residual in
    // FP8 "with almost no loss" (tech report, 2.2), the residual itself stays f32 here. on Vulkan only, which stores
    // them in f16 in the combine before (HC_POST_NORM_F16), quantizes them for the int8 matmul from f16 and reads them in
    // the mean, and from 32 tokens, where that fusion runs. the 395 (14.10): -2.7 to -3.0 percent of the GPU time of a
    // prompt ubatch at 0 / 15K / 118K, KLD 0.015284 against 0.015465 (13.10)
    static const bool hc_f16 = [] {
        const char * env = std::getenv("LLAMA_HC_F16");
        return env == nullptr || std::atoi(env) != 0;
    }();
    if (hc_f16 && il >= 0 && nt >= 32 && cparams.fused_dsv4_hc_pre &&
            std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(model.dev_layer(il))), "Vulkan") == 0) {
        xn = ggml_cast(ctx0, xn, GGML_TYPE_F16);

        // the inputs of the combine before (block output, residual, injection) stay alive until the f16 copy is
        // allocated: half the size of the f32 streams, it fit where they had been freed, and Vulkan then dropped the
        // fusion of the combine with the norm and the copy, whose shader still reads them (395, 13.10: 45 of 93
        // fused). the copy reads only its src[0], extra sources only order the allocation
        if (x->op == GGML_OP_DSV4_HC_POST) {
            const ggml_tensor * w = x->src[2]; // 2*sigmoid(inject/hc): scale <- sigmoid <- scale <- inject
            ggml_tensor * inj = w && w->src[0] && w->src[0]->src[0] ? w->src[0]->src[0]->src[0] : nullptr;
            ggml_tensor * keep[] = { x->src[0], x->src[1], inj };
            int slot = 2;
            for (ggml_tensor * t : keep) {
                if (t != nullptr && slot < GGML_MAX_SRC) {
                    xn->src[slot++] = t;
                }
            }
        }
    }
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = build_lora_mm(w_up, lo);
    cb(gate, "hc_gate", il);

    ggml_tensor * mixed = nullptr;
    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
    } else {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    if (inject) {
        // hc is 4, so the plain [hc_dim, hc] x [hc_dim, nt] product fills 4 rows of a matmul tile.
        // Swapped it is [nt, hc], which backends can run as a mat-vec over nt rows; the transpose
        // back is a few KiB. No lora support here, the hc weights are not adapter targets.
        // swapping puts the weight in the src1 slot, which the backends read as f32 only, so a
        // quantized gamma (the MTP head carries one) keeps the plain order.
        // a few tokens keep it too: swapped, a decode step is a mat-vec over 1 row with 4 columns, one
        // block for 10240 x 4 products. CUDA on the 3090 takes its f32 mat-vec only up to 3 columns and
        // ran it at 67 us, 96 times per token (cuda_op_timing 11.10); plain it is 4 rows of 1 column.
        // LLAMA_HC_INJECT_SWAP_MIN: tokens from which the order is swapped (5, 0 = always)
        static const int64_t swap_min = [] {
            const char * env = std::getenv("LLAMA_HC_INJECT_SWAP_MIN");
            return env ? (int64_t) atoi(env) : (int64_t) 5;
        }();
        if (w_inject->type == GGML_TYPE_F32 && nt >= swap_min) {
            ggml_tensor * inj = ggml_mul_mat(ctx0, xn, w_inject);
            *inject = ggml_cont(ctx0, ggml_transpose(ctx0, inj));
        } else {
            *inject = ggml_mul_mat(ctx0, w_inject, xn);
        }
        cb(*inject, "hc_inject", il);

        // compute the injection right after the mix: xn ([hc_dim, nt] f32) is then freed before the
        // block, not kept until the combine (07.10, the 395 at -ub 4096: compute buffer 8441 -> 7945 MiB)
        ggml_build_forward_expand(gf, mixed);
        ggml_build_forward_expand(gf, *inject);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp_fork::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);

    ggml_tensor * cur = nullptr;
    if (cparams.fused_dsv4_hc_post && il >= 0) {
        // identity comb: every stream adds the same block output, scaled by its own weight
        cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
    } else {
        w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp_fork::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx_fork, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_fork_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);
    // make sure hc_init is in the same graph split as the first layer (-sm tensor); with host embeddings the
    // REPEAT would otherwise stay on the CPU and the device gets hc copies of the embedding (upstream 10f340d1a)
    ggml_build_forward_expand(gf, res_hc);

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = hc_stream_mean(ctx0, res_hc);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        // an unmasked MTP export needs a hidden row for every token, so in that case the gather
        // is deferred until after t_h_nextn is taken below
        const bool gather_now = !cparams.embeddings_nextn || cparams.embeddings_nextn_masked;

        if (il == n_layer - 1 && inp_out_ids && gather_now) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // The MTP head consumes the wide residual, before the head mixer collapses it. Export the
    // combine result itself rather than a reshape of it: a pure view gets no backend assignment
    // from the scheduler, and the readback in llama_context looks one up. It is contiguous, so
    // [n_embd, hc, rows] already has the [n_embd_out, rows] layout the reader expects, and it
    // carries exactly the right rows either way - gathered above when masked, ungathered when not.
    if (cparams.embeddings_nextn) {
        cb(res_hc, "h_nextn", -1);
        res->t_h_nextn = res_hc;

        // deferred from the last layer: collapse to the output rows now that the export is taken
        if (!cparams.embeddings_nextn_masked && inp_out_ids) {
            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for qwen4exp.
//
// The head folds the next token's embedding into the trunk's wide hyper-connection residual,
// runs one trunk-style block over it, and collapses the result with its own mixer before
// reusing the trunk's LM head. The wide post-block residual is exported as t_h_nextn so the
// speculative driver can feed it straight back in for the next draft step.
//
// The block attends densely by default: the trunk's QSA only prunes context past a 2048-token budget, so
// dense is a numerical superset, and drafts are verified by the target either way. LLAMA_MTP_QSA=1 makes it a QSA
// layer like the upstream one (the reference trains it so), to compare acceptance and draft cost at depth.
llama_model_qwen4exp_fork::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP supports a single MTP block");
    ASSERT_EMBD_OR_TOKEN(ubatch);

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int    il    = hparams.n_layer();
    const auto & layer = model.layers[il];

    // a trunk-only file declares the block without its tensors (load_arch_tensors): an error for the context, not an abort
    if (!layer.nextn.eh_proj || !layer.nextn.enorm || !layer.nextn.hnorm || !layer.nextn.hc_head_norm) {
        throw std::runtime_error("qwen4exp: the file has no MTP block tensors, load the MTP head with -md");
    }

    const llama_model & emb_src  = model.tok_embd ? model : qwen4exp_target_model(cparams, model, "token_embd.weight");
    const llama_model & head_src = model.output   ? model : qwen4exp_target_model(cparams, model, "output.weight");

    GGML_ASSERT(head_src.output && "MTP block missing the LM head");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        tok_embd = ggml_get_rows(ctx0, emb_src.tok_embd, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // LLAMA_MTP_QSA (load_arch_hparams): the block attends through QSA over its own indexer cache, as the trunk's
    // full-attention layers do; else densely over a plain KV cache
    const bool mtp_qsa = hparams.dsv4_compress_ratios[il] > 0;

    llm_graph_input_attn_kv *                    inp_attn = nullptr;
    bool                                         rows_only = false; // the attention ran for the output rows only
    llm_graph_input_mem_hybrid *                 inp_hyb  = nullptr;
    const llama_memory_hybrid_idx_fork_context * mctx_hyb = nullptr;
    if (mtp_qsa) {
        inp_hyb  = build_inp_mem_hybrid();
        mctx_hyb = static_cast<const llama_memory_hybrid_idx_fork_context *>(inp_hyb->mctx);

        // the draft memory has no recurrent layer, but its input still has to be allocated
        ggml_build_forward_expand(gf, inp_hyb->get_recr()->s_copy);
    } else {
        inp_attn = build_attn_inp_kv();
    }

    // grouped RMSNorm over the wide stream: normalise each hc stream, then scale with the
    // [n_embd, hc] gamma, exactly as build_hc_mix does
    ggml_tensor * h_norm = ggml_mul(ctx0, ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps), layer.nextn.hnorm);
    cb(h_norm, "mtp_hnorm", il);

    // the token embedding is shared across the streams, so broadcast it to hc copies
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // eh_proj holds fc_embedding and fc_hidden side by side, so this one matmul is
    // fc_embedding @ e_norm + fc_hidden @ h_norm, applied to each stream independently.
    // Keeping the streams distinct here is the point of the hyper-connection residual:
    // pooling them before the projection would throw that away.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    // one matmul over hc*n_tokens columns: as n_tokens batches of hc columns, Vulkan runs a full
    // tile per batch and sizes the src1 copy for it (8x the data, 2.5 GiB at -ub 8192)
    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj,
            ggml_reshape_2d(ctx0, concat, 2*n_embd, hc*n_tokens), layer.nextn.eh_proj_s);
    res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, n_tokens);
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    if (mtp_qsa) {
        // a catch-up over prompt rows with no output (common_speculative process()): K, V and the indexer keys of
        // every row are stored, nothing attends, as the dense branch below does for its rows. attending for all rows
        // took the 395 prompt down 4 percent at 80-120K and 7 with a budget of 8192 (13.10 night, chats).
        // LLAMA_MTP_ROWS_ALL=1: attend for every row
        static const bool rows_all_qsa = [] {
            const char * env = std::getenv("LLAMA_MTP_ROWS_ALL");
            return env != nullptr && std::atoi(env) != 0;
        }();
        if (!rows_all_qsa && !cparams.training && inp_out_ids != nullptr && inp_out_ids->ne[0] == 0) {
            build_layer_attn(inp_hyb->get_attn(), mctx_hyb, cur, inp_pos, sections, il, true);
            // no row: the rows below are empty and nothing reads the attention
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            rows_only = true;
        } else {
            cur = build_layer_attn(inp_hyb->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }
    } else {
        // dense self-attention, mirroring the trunk's full-attention branch
        const int64_t n_embd_head = hparams.n_embd_head_v();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

        // wq holds [q|gate] interleaved per head
        ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur_full, "mtp_Qcur_full", il);

        ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "mtp_Qcur_normed", il);

        ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
        gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
        cb(gate, "mtp_gate", il);

        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "mtp_Kcur_normed", il);

        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        cb(Vcur, "mtp_Vcur", il);

        // IMRoPE, same convention and freq_base as the trunk
        Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "mtp_Qcur", il);
        cb(Kcur, "mtp_Kcur", il);

        const float kq_scale = hparams.f_attention_scale == 0.0f
                ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        // the K and V of a row depend on the target's hidden state and the next token only, not on the attention
        // output, so a batch that needs fewer outputs than rows (the catch-up over a prompt: none, or the last row)
        // stores K and V of every row and attends for the output rows alone; the MoE and the head already run on
        // those. at a 118K prompt on the 395 the draft context took 55 s of attention for all rows (09.10).
        // LLAMA_MTP_ROWS_ALL=1: attend for every row, as before
        static const bool rows_all = [] {
            const char * env = std::getenv("LLAMA_MTP_ROWS_ALL");
            return env != nullptr && std::atoi(env) != 0;
        }();
        ggml_tensor * kq_mask = inp_attn->get_kq_mask();
        const bool manual_ok = !cparams.training && inp_attn->self_k_rot == nullptr && inp_attn->self_v_rot == nullptr &&
            kq_mask != nullptr && kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1 && ggml_is_contiguous(kq_mask);

        // LLAMA_MTP_WINDOW=<cells> (16384; 0: the whole context): the draft attends to the last <cells> cells of its KV
        // only. its dense attention reads the whole context per draft step: on the 395 the step took 3.7 ms at depth 0
        // and 5.6 at ~195K (05, 15.10). the drafts are verified by the target, a window changes the acceptance only. the
        // cells of the draft memory follow the positions (one sequence, filled in order), so the last ones are the
        // nearest. the 395 (15.10 night, chats to 213K): 16384 against the whole context TG 38.64 / 36.39 / 36.84
        // against 37.08 / 34.76 / 35.37 at 120-160K / 160-200K / 200K+, acceptance 0.719 against 0.723
        // with several slots (or LLAMA_MTP_WINDOW_KV=1) the cache itself is the window (mtp_window_kv) and its mask
        // has the positions; its cells are reused out of order, so the last ones are not the nearest
        const int64_t mtp_window = llama_model_qwen4exp_fork::mtp_window();
        uint32_t       wkv_size = 0;
        uint32_t       wkv_swa  = 0;
        llama_swa_type wkv_type = LLAMA_SWA_TYPE_NONE;
        const bool window_kv = llama_model_qwen4exp_fork::mtp_window_kv(cparams, wkv_size, wkv_swa, wkv_type);
        ggml_tensor * k_all = inp_attn->mctx->get_k(ctx0, il);
        ggml_tensor * v_all = inp_attn->mctx->get_v(ctx0, il);
        const bool windowed = mtp_window > 0 && !window_kv && manual_ok && cparams.n_seq_max == 1 && kq_mask->ne[0] > mtp_window &&
            k_all->ne[2] == kq_mask->ne[0] && v_all->nb[1] <= v_all->nb[2];

        rows_only = manual_ok && ((!rows_all && inp_out_ids != nullptr && inp_out_ids->ne[0] < n_tokens) || windowed);

        if (rows_only) {
            ggml_build_forward_expand(gf, Vcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));

            const int64_t n_out = inp_out_ids ? inp_out_ids->ne[0] : n_tokens;

            ggml_tensor * q_out = ggml_reshape_2d(ctx0, Qcur, n_embd_head*n_head, n_tokens);
            if (inp_out_ids) {
                q_out = ggml_get_rows(ctx0, q_out, inp_out_ids);
                gate  = ggml_get_rows(ctx0, gate, inp_out_ids);
            }

            if (n_out > 0) {
                // the mask rows of the output tokens; get_rows of an f16 source gives f32, the attention reads f16
                ggml_tensor * mask_out = ggml_reshape_2d(ctx0, kq_mask, kq_mask->ne[0], kq_mask->ne[1]);
                if (inp_out_ids) {
                    mask_out = ggml_get_rows(ctx0, mask_out, inp_out_ids);
                }
                ggml_tensor * k = k_all;
                ggml_tensor * v = v_all;
                if (windowed) {
                    const int64_t n_kv = kq_mask->ne[0];
                    const int64_t off  = n_kv - mtp_window;
                    k = ggml_view_4d(ctx0, k_all, k_all->ne[0], k_all->ne[1], mtp_window, k_all->ne[3], k_all->nb[1], k_all->nb[2], k_all->nb[3], off*k_all->nb[2]);
                    v = ggml_view_4d(ctx0, v_all, v_all->ne[0], v_all->ne[1], mtp_window, v_all->ne[3], v_all->nb[1], v_all->nb[2], v_all->nb[3], off*v_all->nb[2]);
                    mask_out = ggml_cont(ctx0, ggml_view_2d(ctx0, mask_out, mtp_window, n_out, mask_out->nb[1], off*ggml_element_size(mask_out)));
                }
                if (mask_out->type != GGML_TYPE_F16) {
                    mask_out = ggml_cast(ctx0, mask_out, GGML_TYPE_F16);
                }

                cur = build_attn_mha(ggml_reshape_3d(ctx0, q_out, n_embd_head, n_head, n_out),
                        k, v, nullptr, mask_out, nullptr, nullptr, 0, kq_scale, il);
            } else {
                // no output row: the rows below are empty and nothing reads the attention
                cur = q_out;
            }
        } else {
            cur = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        }
        cb(cur, "mtp_attn_pregate", il);

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        cb(cur, "mtp_attn_gated", il);

        cur = build_lora_mm(layer.wo, cur, layer.wo_s);
        cb(cur, "mtp_attn_out", il);
    }

    if (inp_out_ids) {
        if (!rows_only) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    // MoE, identical to the trunk's build_layer_ffn
    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
            &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    // The next draft step re-enters here, so export the wide stream before it is collapsed.
    // As in the trunk, export the combine result rather than a reshape view of it.
    cb(res_hc, "h_nextn", -1);
    res->t_h_nextn = res_hc;

    // the head's own mixer collapses the streams and doubles as the output norm
    cur = build_hc_mix(res_hc,
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // deliberately no res->t_embd: it would be n_embd wide while the context sizes its
    // embedding buffer by n_embd_out (the wide stream). The driver reads t_h_nextn instead.

    cur = build_lora_mm(head_src.output, cur, head_src.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp_fork::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp_fork::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
class llama_model_qwen4exp_fork::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_fork_context * mctx, uint32_t ratio, bool blk_bias, bool causal_attn) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias), causal_attn(causal_attn) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);

        if (bypass) {
            // the indexer was skipped, so nothing but k_idxs was ever built or allocated
            return;
        }

        // keys_only scores nothing: no bias and no block tables were built
        if (!keys_only) {
            mctx->set_input_qsa(cell_blk, blk_cells, bias, ubatch, ratio, blk_bias, causal_attn);
        }
        mctx->set_input_qsa_dirty(dirty_cells, dirty_pos, dirty_rows, ratio);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_fork_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const int64_t n_blocks = (n_kv + ratio - 1)/ratio;

        bool res = true;

        res &= params.ubatch.n_tokens % n_stream == 0;

        res &= k_idxs->ne[0] == params.ubatch.n_tokens;

        // the graph keeps the indexer only while the budget fails to cover the cache
        const int64_t width     = std::min<int64_t>(n_kv, budget);
        const int64_t width_blk = std::min<int64_t>(n_blocks, (width + ratio - 1)/ratio);

        res &= bypass == (blk_bias ? width_blk >= n_blocks : width >= n_kv);

        if (bypass) {
            return res;
        }

        if (cell_blk) {
            res &= cell_blk->ne[0] == n_kv;
            res &= cell_blk->ne[1] == n_stream;
        }

        if (blk_cells) {
            res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
            res &= blk_cells->ne[1] == n_stream;
        }

        if (bias) {
            res &= bias->ne[0] == (blk_bias ? n_blocks : n_kv);
            res &= bias->ne[1] == params.ubatch.n_tokens/n_stream;
        }

        // how many block keys this pass has to rebuild decides the shape of the dirty list
        res &= dirty_rows->ne[0] == (int64_t) mctx->qsa_prepare(&params.ubatch, ratio, blk_bias).n_dirty;
        res &= dirty_rows->ne[1] == n_stream;

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream], only without blk_bias
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream], only with blk_bias
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    // the blocks whose pooled key this pass rebuilds, padded to a size that holds still
    ggml_tensor * dirty_cells = nullptr; // I32 [ratio*n_dirty, n_stream]
    ggml_tensor * dirty_pos   = nullptr; // I32 [4*n_dirty*n_stream]
    ggml_tensor * dirty_rows  = nullptr; // I32 [n_dirty, n_stream], rows of the pool view

    // when the budget covers the cache the indexer is skipped and only k_idxs is built
    bool    bypass = false;
    int64_t budget = 0;

    // the keys of every row are stored and the pooled ones rebuilt, nothing is scored (see build_layer_attn)
    bool    keys_only = false;

    const llama_memory_hybrid_idx_fork_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;

    // this is fixed for the graph's lifetime, as causal_attn is part of the reuse key (llm_graph_params::allow_reuse)
    const bool causal_attn;
};

// the ubatches the QSA mask values can come from positions for: tokens only. the KQ mask of an M-RoPE model applies its
// 2D rule (is_pos_2d is true for every ubatch of such a model) only where a cell and the token share the temporal
// position; a text token shares it with its own cell alone (its x and y equal it), which the rule keeps visible, while
// the tokens of an image share one temporal position
// every stream of a split cache holds one sequence; a unified one can take the tokens of several server slots in one
// ubatch, and then each sequence gets a range of positions of its own (qsa_pos_multi, 23.10), unless a cell is held by
// two sequences (llama_memory_hybrid_idx_fork::qsa_pos_multi_ok: the dense mask then)
static bool qsa_mask_pos_ubatch(const llama_ubatch & ubatch) {
    return ubatch.token != nullptr && ubatch.embd == nullptr && !ubatch.is_mixed();
}

static bool qsa_pos_multi(const llama_cparams & cparams, const llama_ubatch & ubatch) {
    return cparams.kv_unified && ubatch.n_seqs_unq > 1;
}

// the mask values of the cells QSA picks, from positions instead of the dense KQ mask (see build_attn_qsa): the position
// of every attention cell of a stream's sequence (+inf for the rest), the position of every token, and the -inf that
// fills the mask before the picked cells get their values
class llama_model_qwen4exp_fork::llm_graph_input_qsa_pos : public llm_graph_input_i {
public:
    explicit llm_graph_input_qsa_pos(const llama_memory_hybrid_idx_fork_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_qsa_pos() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->set_input_qsa_pos(cell_pos, tok_pos, tok_half, ubatch);

        // with LLAMA_QSA_FA_LIST no mask is filled and nothing reads the -inf, so it is never allocated
        if (neg_inf->buffer) {
            const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
            ggml_backend_tensor_set(neg_inf, &drop, 0, sizeof(drop));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_fork_context *>(params.mctx);

        const int64_t n_stream = mctx->get_n_stream();

        bool res = qsa_mask_pos_ubatch(params.ubatch); // an image ubatch needs the dense mask

        // the ranges of several sequences in one stream take one more input and two more ops
        const bool multi = qsa_pos_multi(params.cparams, params.ubatch);
        res &= multi == (tok_half != nullptr);
        res &= !multi || mctx->qsa_pos_multi_ok(&params.ubatch);

        res &= params.ubatch.n_tokens % n_stream == 0;
        res &= cell_pos->ne[1] == (int64_t) mctx->get_attn()->get_n_kv();
        res &= cell_pos->ne[3] == n_stream;
        res &= tok_pos->ne[2] == params.ubatch.n_tokens/n_stream;

        return res;
    }

    ggml_tensor * cell_pos = nullptr; // F32 [1, n_kv, 1, n_stream]
    ggml_tensor * tok_pos  = nullptr; // F32 [1, 1, n_tokens/n_stream, n_stream]; the mid of a range with several sequences
    ggml_tensor * tok_half = nullptr; // F32 [1, 1, n_tokens/n_stream, n_stream]: the half width, only with several sequences
    ggml_tensor * neg_inf  = nullptr; // F16 [1, 1, 1, 1]

    const llama_memory_hybrid_idx_fork_context * mctx;
};

// the lightning indexer for the block scores. its CUDA and Vulkan kernels take a head size of 128.
// 08.10, 395 (Vulkan): a decode step at 122880 0.77 ms faster than MUL_MAT_HEADSUM, a 4096-token prompt 6 percent
// slower there; 3090 (CUDA): the prompt the same, TG +1.7 percent. so Vulkan takes it up to LLAMA_QSA_LID_TOKENS
// tokens per stream (default 64: decode and draft checks), every other backend always.
// 12.10: Vulkan runs a prompt of 4 heads in one stream on the fused head matmul inside the op (ggml_vk_fork_lid_headsum),
// with one score and an f16 bias instead of the head scores and an f32 bias. on the test model that took less memory,
// on the 395 the compute buffer of the real model went 5979 -> 10495 MiB and the prompt and decode got slower, cause
// not found yet (GGML_ALLOC_PEAK); so the default stays 64, LLAMA_QSA_LID_TOKENS=1000000 takes the op at every size.
// LLAMA_QSA_LID=1: always, 0: never
static bool qsa_use_lid(int64_t idx_dim, int64_t n_idx_h, int64_t n_tps, int64_t n_stream, bool is_vulkan) {
    GGML_UNUSED(n_idx_h);
    GGML_UNUSED(n_stream);
    static const int mode = [] {
        const char * env = std::getenv("LLAMA_QSA_LID");
        return env == nullptr ? -1 : std::atoi(env) != 0;
    }();
    static const int64_t vk_tokens = [] {
        const char * env = std::getenv("LLAMA_QSA_LID_TOKENS");
        return env == nullptr ? (int64_t) 64 : (int64_t) std::atoll(env);
    }();
    if (idx_dim != 128 || mode == 0) {
        return false;
    }
    return mode == 1 || !is_vulkan || n_tps <= vk_tokens;
}

// LLAMA_MTP_QSA_TOPK: the token budget of a QSA MTP block, unset or 0 for the trunk's indexer_top_k. one layer, so a
// wider budget costs little per draft step, and the dense block drafted better at depth than the QSA one with 2048
// (09.10 on the 395 at 118K: acceptance 0.63 / 0.41 / 0.29 by position against 0.59 / 0.34 / 0.23)
static int64_t qsa_top_k_tokens(const llama_hparams & hp, int il) {
    static const int64_t mtp = [] {
        const char * env = std::getenv("LLAMA_MTP_QSA_TOPK");
        return env ? (int64_t) std::atoll(env) : (int64_t) 0;
    }();
    return il >= (int) hp.n_layer() && mtp > 0 ? mtp : (int64_t) hp.indexer_top_k;
}

// LLAMA_MTP_QSA_REUSE=0: a QSA MTP block runs its indexer at every draft step (see llama_context::set_qsa_reuse)
static bool qsa_mtp_reuse_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("LLAMA_MTP_QSA_REUSE");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

ggml_tensor * llama_model_qwen4exp_fork::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_fork_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il,
        bool                                    keys_only) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // the reference returns indexer_top_k + compress_ratio - 1 cells: whole blocks plus the tail
    const int64_t top_k_tok = qsa_top_k_tokens(hparams, il);
    const int64_t width     = std::min<int64_t>(n_kv, top_k_tok + r - 1);
    const int64_t width_blk = std::min<int64_t>(n_blocks, (width + r - 1)/r);

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask, so it opts out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        !hparams.use_alibi;

    // when the budget already covers the cache, top_k names every cell and set_rows restores the
    // whole mask, which leaves the plain causal mask: the scoring below cannot change the result.
    // only k_idxs is built then - a tensor no node reads is never allocated, so building the rest
    // would leave set_input writing into null buffers.
    const bool bypass = blk_bias ? width_blk >= n_blocks : width >= n_kv;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias, cparams.causal_attn);

        qsa->k_idxs = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->bypass = bypass;
        qsa->budget = top_k_tok + r - 1;
        qsa->keys_only = keys_only;

        if (!bypass) {
            // only one of the two is ever read: the per-block selection turns chosen blocks back
            // into cells, the per-cell one spreads a block score over its cells. an input no node
            // reads is never allocated, so building both would leave set_input writing into null
            if (keys_only) {
                // nothing is selected or scored
            } else if (blk_bias) {
                qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);

                ggml_set_input(qsa->blk_cells);
            } else {
                qsa->cell_blk = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);

                ggml_set_input(qsa->cell_blk);
            }

            // the lightning indexer takes the per-block bias as its f16 mask, see the scores below
            if (!keys_only) {
                ggml_backend_dev_t dev_l = model.dev_layer(il);
                const bool vk_l = std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev_l)), "Vulkan") == 0;
                qsa->bias = ggml_new_tensor_3d(ctx0, blk_bias && qsa_use_lid(idx_dim, n_idx_h, n_tps, n_stream, vk_l) ? GGML_TYPE_F16 : GGML_TYPE_F32,
                        blk_bias ? n_blocks : n_kv, n_tps, n_stream);

                ggml_set_input(qsa->bias);
            }

            // the pooled block keys are kept across passes, so only the moved ones are rebuilt
            const int64_t n_dirty = mctx_hyb->qsa_prepare(&ubatch, (uint32_t) r, blk_bias).n_dirty;

            qsa->dirty_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_dirty, n_stream);
            qsa->dirty_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_dirty*n_stream);
            qsa->dirty_rows  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_dirty, n_stream);

            ggml_set_input(qsa->dirty_cells);
            ggml_set_input(qsa->dirty_pos);
            ggml_set_input(qsa->dirty_rows);
        }

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // the per-block selection: the top width_blk blocks of each token, turned back into cells
    auto select_blocks = [&](ggml_tensor * score) {
        ggml_tensor * top_k_blk = ggml_top_k(ctx0, score, width_blk);

        // blk_cells row b lists the cells of block b, so this turns the chosen blocks into cells
        // an i32 source keeps get_rows in i32, which is what the mask build wants
        ggml_tensor * gathered = ggml_get_rows(ctx0,
                ggml_reshape_3d(ctx0, inp->blk_cells, r, n_blocks, n_stream),
                ggml_reshape_2d(ctx0, top_k_blk, width_blk*n_tps, n_stream));

        // [r, width_blk*n_tps, n_stream] -> [n_top_k, n_batch, 1, n_stream], as build_attn_qsa reads it
        ggml_tensor * top_k = ggml_reshape_4d(ctx0, gathered, r*width_blk, n_tps, 1, n_stream);

        cb(top_k, "indexer_top_k", il);

        // into the graph here, so the indexer runs and frees its block scores (n_blocks x n_tps f32, 1 GiB at 262144 x
        // 4096) before the attention builds its mask. reached only through the attention, the selection came after
        // the -inf fill of the mask and both were alive at once: the 395 compute buffer took +4.5 GiB with the
        // lightning indexer (08.10 and 12.10), where the matmul path had its score in the graph already
        ggml_build_forward_expand(gf, top_k);

        return top_k;
    };

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // the indexer keys still have to reach the cache, so the bypass returns only after cpy_k
    if (bypass) {
        return nullptr;
    }

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // the pooled keys of a block only change where its cells do, so the whole set is kept in the
    // memory module and this pass rebuilds the dirty ones. set_input pads the list to a size that
    // holds still, so the graph stays reusable
    const int64_t n_dirty = inp->dirty_rows->ne[0];

    // gathers per stream: dirty_cells row s indexes stream s's own cells
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->dirty_cells);
    members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_dirty, n_stream);

    // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows.
    // the slices stay views: add takes its own strides on every backend, while materializing them
    // costs a read and a write of the whole gather per slice
    std::vector<ggml_tensor *> slices(r);
    for (int64_t i = 0; i < r; ++i) {
        slices[i] = ggml_view_3d(ctx0, members, idx_dim, n_dirty, n_stream,
                members->nb[2], members->nb[3], i*members->nb[1]);

        // the views go into the graph first, so the adds below end up next to each other.
        // vulkan folds a run of adds into one multi_add, and a view in between cuts the run
        ggml_build_forward_expand(gf, slices[i]);
    }

    ggml_tensor * vals = slices[0];
    for (int64_t i = 1; i < r; ++i) {
        vals = ggml_add(ctx0, vals, slices[i]);
    }

    // a one-member block leaves vals a view, and scale wants a contiguous source
    if (!ggml_is_contiguous(vals)) {
        vals = ggml_cont(ctx0, vals);
    }

    vals = ggml_scale(ctx0, vals, 1.0f/(float) r);
    cb(vals, "indexer_k_pooled", il);

    // count blocks along ne1: rms_norm launches gridDim.y = ne2, capped at 65535, and 262144/4 = 65536
    vals = ggml_reshape_3d(ctx0, vals, idx_dim, n_dirty*n_stream, 1);
    vals = build_norm(vals, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
    vals = ggml_reshape_3d(ctx0, vals, idx_dim, 1, n_dirty*n_stream);
    vals = ggml_rope_multi(ctx0, vals, inp->dirty_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    // store the rebuilt keys, then read the whole range back through the store: the read is a
    // view of it, so nothing below can be scheduled before the write
    ggml_tensor * pool = mctx_hyb->get_pool(ctx0, il);
    GGML_ASSERT(pool != nullptr);

    const int64_t n_pool = pool->ne[1];

    pool = ggml_set_rows(ctx0,
            ggml_reshape_2d(ctx0, pool, idx_dim, n_pool*n_stream),
            ggml_reshape_2d(ctx0, vals, idx_dim, n_dirty*n_stream),
            ggml_reshape_1d(ctx0, inp->dirty_rows, n_dirty*n_stream));

    // blocks past the live ones keep an old key. they are never picked on their score: the bias
    // hides a block with no cells, and always takes one whose cells no block pools
    cb(pool, "indexer_k", il);

    if (keys_only) {
        ggml_build_forward_expand(gf, pool);
        return nullptr;
    }

    // an MTP block keeps the block scores of the first draft step of a round (one token per stream, without the
    // bias), and the steps after it (cparams.qsa_reuse, set by the speculative driver) select from them again under
    // their own bias instead of running the indexer: the reference reuses its top-k across MTP steps, whose queries
    // come from the head's own hidden state rather than the target's. the bias of a step still brings in the block
    // of its own position. a first step that skipped the indexer (the budget covered the cache) leaves the scores of
    // an older round, finite, for at most the steps of one round
    ggml_tensor * saved = il >= (int) hparams.n_layer() && n_tps == 1 && blk_bias && qsa_mtp_reuse_enabled() ?
        mctx_hyb->get_score(ctx0, il) : nullptr;
    if (saved) {
        saved = ggml_view_3d(ctx0, saved, n_blocks, 1, n_stream, saved->nb[1], saved->nb[1], 0);
    }

    if (saved && cparams.qsa_reuse) {
        // nothing below reads the pool in this pass, but the rebuilt keys still have to reach it
        ggml_build_forward_expand(gf, pool);

        ggml_tensor * bias_f32 = inp->bias->type == GGML_TYPE_F32 ? inp->bias : ggml_cast(ctx0, inp->bias, GGML_TYPE_F32);
        ggml_tensor * score    = ggml_add(ctx0, saved, bias_f32);
        cb(score, "indexer_score_reused", il);

        return select_blocks(score);
    }

    // the scores of a first step go to the store before the bias is added
    auto save_scores = [&](ggml_tensor * score) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, score, saved));
    };

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream);

    ggml_backend_dev_t dev = model.dev_layer(il);

    // the lightning indexer computes sum_h relu(q_h.k) + mask in one op: one [n_blocks, n_tokens] f32 output and an
    // f16 bias, where the matmul paths below hold a score per head (a head score is 1 GiB at n_kv 262144 with a
    // 4096-token ubatch). the reference sums the heads unweighted, so the weights are 1 (upstream scales by
    // 1/sqrt(d), which keeps the ranking). LLAMA_QSA_LID=0: the matmul paths
    ggml_tensor * bias = inp->bias;
    if (blk_bias && bias->type == GGML_TYPE_F16) {
        ggml_tensor * q4 = ggml_reshape_4d(ctx0, q, idx_dim, n_idx_h, n_tps, n_stream);
        ggml_tensor * k4 = ggml_view_4d(ctx0, pool, idx_dim, 1, n_blocks, n_stream,
                pool->nb[1], pool->nb[1], n_pool*pool->nb[1], 0);
        ggml_tensor * w  = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_idx_h, n_tps, 1, n_stream), 1.0f);
        ggml_tensor * m  = ggml_reshape_4d(ctx0, bias, n_blocks, n_tps, 1, n_stream);

        // the scores to keep come without the bias: a zero mask, the bias added after
        if (saved) {
            m = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_blocks, n_tps, 1, n_stream), 0.0f);
        }

        ggml_tensor * lid = ggml_lightning_indexer(ctx0, q4, k4, w, m); // [n_blocks, n_tps, 1, n_stream]

        if (ggml_backend_dev_supports_op(dev, lid)) {
            ggml_tensor * score = ggml_reshape_3d(ctx0, lid, n_blocks, n_tps, n_stream);
            if (saved) {
                save_scores(score);
                score = ggml_add(ctx0, score, ggml_cast(ctx0, bias, GGML_TYPE_F32));
            }
            cb(score, "indexer_score_blocks", il);

            return select_blocks(score);
        }

        // a device without the op reads the same bias in f32 (0, -inf and +inf are exact in f16)
        bias = ggml_cast(ctx0, bias, GGML_TYPE_F32);
    }

    // Vulkan fuses split heads into one matmul with the sum at its store. Other backends batch the heads.
    const bool is_vulkan = std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "Vulkan") == 0;
    bool split_heads = is_vulkan;
    if (const char * env = std::getenv("LLAMA_QSA_SPLIT_HEADS")) {
        if (std::strcmp(env, "0") == 0 || std::strcmp(env, "1") == 0) {
            split_heads = std::strcmp(env, "1") == 0;
        }
    }

    // Use the full pool size so reserve and execution choose the same graph at every KV depth.
    const size_t max_size = ggml_backend_buft_get_max_size(ggml_backend_dev_buffer_type(dev));
    const size_t score_row_size = ggml_row_size(GGML_TYPE_F32, n_pool);
    if ((uint64_t) n_idx_h*n_tps*n_stream > max_size / score_row_size) {
        split_heads = true;
    }

    std::vector<ggml_tensor *> ks(n_stream);
    for (int64_t s = 0; s < n_stream; ++s) {
        ks[s] = ggml_view_2d(ctx0, pool, idx_dim, n_blocks, pool->nb[1], s*n_pool*pool->nb[1]);
    }

    ggml_tensor * score = nullptr;
    if (!split_heads) {
        for (int64_t s = 0; s < n_stream; ++s) {
            ggml_tensor * qs = ggml_view_2d(ctx0, q, idx_dim, n_idx_h*n_tps, q->nb[1], s*q->nb[2]);
            ggml_tensor * sc = ggml_mul_mat(ctx0, ks[s], qs);
            if (n_stream > 1) {
                sc = ggml_reshape_3d(ctx0, sc, n_blocks, n_idx_h*n_tps, 1);
            }
            score = score ? ggml_concat(ctx0, score, sc, 2) : sc;
        }
        score = ggml_relu(ctx0, score);
        score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);

        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * head = ggml_view_3d(ctx0, score, n_blocks, n_tps, n_stream, score->nb[2], score->nb[3], h*score->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, head) : ggml_cont(ctx0, head);
        }
        score = summed;
    } else {
        // each head joins the sum right after it is computed: the allocator then holds two head scores at most, not
        // all of them. a head score is 1 GiB at n_kv 262144 with a 4096-token ubatch, and with all four alive the
        // compute buffer of the 395 grew to 10.5 GB and put the device at the edge of its memory (06.10)
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * head = nullptr;

            for (int64_t s = 0; s < n_stream; ++s) {
                ggml_tensor * qs = ggml_view_2d(ctx0, q, idx_dim, n_tps, n_idx_h*q->nb[1], s*q->nb[2] + h*q->nb[1]);
                ggml_tensor * sc = ggml_mul_mat(ctx0, ks[s], qs);

                if (n_stream > 1) {
                    sc = ggml_reshape_3d(ctx0, sc, n_blocks, n_tps, 1);
                }
                head = head ? ggml_concat(ctx0, head, sc, 2) : sc;
            }

            head  = ggml_relu(ctx0, head);
            score = score ? ggml_add(ctx0, score, head) : head;
            ggml_build_forward_expand(gf, score);
        }
    }
    cb(score, "indexer_score", il);

    // the per-block bias already says which blocks the query can see at all, so the selection can
    // run on the block scores. expanding them to a score per cell first would cost n_kv floats per
    // token instead of n_blocks, and the same again for the mask add - the bulk of the indexer.
    if (blk_bias) {
        if (saved) {
            save_scores(score);
        }
        score = ggml_add(ctx0, score, bias);
        cb(score, "indexer_score_blocks", il);

        // the head scores, their sum and the bias add in a row in the graph: a backend can do them in one matmul
        ggml_build_forward_expand(gf, score);

        return select_blocks(score);
    }

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
    ggml_tensor * expanded = ggml_get_rows(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
    expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

    expanded = ggml_add(ctx0, expanded, inp->bias);
    cb(expanded, "indexer_score_tokens", il);

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

static void qsa_pack_kv(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * k    = dst->src[0];
    const ggml_tensor * v    = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && ids->type == GGML_TYPE_I32);

    const int64_t width = ids->ne[0];
    const int64_t n_tps = ids->ne[1];
    const size_t k_bytes = k->ne[0]*k->ne[1]*sizeof(ggml_fp16_t);
    const size_t v_bytes = v->ne[0]*v->ne[1]*sizeof(ggml_fp16_t);
    const ggml_fp16_t masked = ggml_fp32_to_fp16(-INFINITY);
    std::vector<int32_t> cells(width);

    for (int64_t i = ith; i < n_tps*ids->ne[3]; i += nth) {
        const int64_t t = i % n_tps;
        const int64_t s = i / n_tps;
        for (int64_t j = 0; j < width; ++j) {
            cells[j] = *(const int32_t *) ((const char *) ids->data + j*ids->nb[0] + t*ids->nb[1] + s*ids->nb[3]);
        }
        // The dense mask visits a cell once, in cell order, even if top_k names it twice.
        std::sort(cells.begin(), cells.end());
        for (int64_t j = 0; j < dst->ne[1]; ++j) {
            char * row = (char *) dst->data + j*dst->nb[1] + t*dst->nb[2] + s*dst->nb[3];
            if (j >= width) {
                memset(row, 0, k_bytes + v_bytes);
                memcpy(row + k_bytes + v_bytes, &masked, sizeof(ggml_fp16_t));
                continue;
            }
            const int32_t cell = cells[j];
            GGML_ASSERT(cell >= 0 && cell < k->ne[2]);
            memcpy(row,           (const char *) k->data + cell*k->nb[2] + s*k->nb[3], k_bytes);
            memcpy(row + k_bytes, (const char *) v->data + cell*v->nb[2] + s*v->nb[3], v_bytes);
            const void * bias = j > 0 && cell == cells[j - 1] ? (const void *) &masked :
                (const char *) mask->data + cell*mask->nb[0] + t*mask->nb[1] + s*mask->nb[3];
            memcpy(row + k_bytes + v_bytes, bias, sizeof(ggml_fp16_t));
        }
    }
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp_fork::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_fork_context * mctx_hyb,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        bool                      kv_host,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    ggml_tensor * k_store;
    ggml_tensor * v_store;
    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        k_store = mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il);
        v_store = mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il);
        if (kv_host) {
            ggml_backend_sched_set_tensor_backend(sched, k_store, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, v_store, backend_cpu);
        }
        ggml_build_forward_expand(gf, k_store);
        ggml_build_forward_expand(gf, v_store);
    }

    if (kv_host && n_tokens < 32) {
        ggml_tensor * k = mctx_cur->get_k(ctx0, il);
        ggml_tensor * v = mctx_cur->get_v(ctx0, il);
        k = ggml_view_4d(ctx0, k_store, k->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], k->view_offs);
        v = ggml_view_4d(ctx0, v_store, v->ne[0], v->ne[1], v->ne[2], v->ne[3], v->nb[1], v->nb[2], v->nb[3], v->view_offs);
        GGML_ASSERT(k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        GGML_ASSERT(k->nb[1] == k->ne[0]*sizeof(ggml_fp16_t) && v->nb[1] == v->ne[0]*sizeof(ggml_fp16_t));

        const int64_t width = GGML_PAD(top_k->ne[0], 256);
        const int64_t n_tps = top_k->ne[1];
        const int64_t ns    = top_k->ne[3];
        const int64_t nk    = k->ne[0]*k->ne[1];
        const int64_t nv    = v->ne[0]*v->ne[1];
        ggml_tensor * args[] = { k, v, top_k, inp->get_kq_mask() };
        ggml_tensor * pack = ggml_custom_4d(ctx0, GGML_TYPE_F16, GGML_PAD(nk + nv + 1, 64), width, n_tps, ns,
                args, 4, qsa_pack_kv, GGML_N_TASKS_MAX, nullptr);
        cb(pack, "qsa_kv_pack", il);

        auto extract = [&](int64_t ne0, size_t offset) {
            ggml_tensor * t = ggml_view_4d(ctx0, pack, ne0, width, n_tps, ns, pack->nb[1], pack->nb[2], pack->nb[3], offset);
            t = ggml_cont(ctx0, t);
            ggml_backend_sched_set_tensor_backend(sched, t, backend_cpu);
            return t;
        };
        ggml_tensor * mask = extract(1, (nk + nv)*sizeof(ggml_fp16_t));
        k = ggml_reshape_4d(ctx0, extract(nk, 0), k->ne[0], k->ne[1], width, n_tps*ns);
        v = ggml_reshape_4d(ctx0, extract(nv, nk*sizeof(ggml_fp16_t)), v->ne[0], v->ne[1], width, n_tps*ns);
        mask = ggml_reshape_4d(ctx0, mask, width, 1, 1, n_tps*ns);

        ggml_tensor * cur = build_attn_mha(q_cur, k, v, nullptr, mask, nullptr, nullptr, 0, kq_scale, il);
        cb(cur, "kqv_out", il);
        return cur;
    }

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    ggml_tensor * kq_mask = inp->get_kq_mask();
    const int64_t n_kv_m  = kq_mask->ne[0];
    const int64_t n_tps_m = kq_mask->ne[1];
    const int64_t n_str_m = kq_mask->ne[3];

    // the mask values of the selected cells. from positions (LLAMA_QSA_MASK_POS, default 1): a cell is visible to a token
    // when the token's sequence holds it and its position is not above the token's, which is what the KQ mask says
    // for a ubatch of tokens (qsa_mask_pos_ubatch), causal, without ALiBi or SWA; several sequences in one stream (the
    // slots of a unified cache) through a range of positions per sequence (qsa_pos_multi). the dense mask then is never
    // built: it took n_kv x n_batch f16 twice, pinned on the host and as its copy on the device (2 GiB each at
    // 262144 x 4096, the 395 11.10), and as much host time to fill and copy per ubatch. otherwise, and with 0, the
    // values are read out of the KQ mask
    static const bool pos_mode = [] {
        const char * env = std::getenv("LLAMA_QSA_MASK_POS");
        return env == nullptr || std::atoi(env) != 0;
    }();
    const bool pos_multi = qsa_pos_multi(cparams, ubatch);
    const bool from_pos  = pos_mode && cparams.causal_attn && !hparams.use_alibi && hparams.swa_type == LLAMA_SWA_TYPE_NONE &&
        qsa_mask_pos_ubatch(ubatch) && (!pos_multi || mctx_hyb->qsa_pos_multi_ok(&ubatch));

    // fork: on Vulkan the attention can take the selected cells and their values as a list (GGML_OP_FLASH_ATTN_LIST):
    // no -inf mask of n_kv x n_batch (2 GiB at 262144 x 4096), no set_rows into it and no pre-pass of the sparse path
    // that finds the cells in it again; on the 395 (13.10, 122880, ubatch 4096) the pre-pass took ~90 ms and the fill
    // of the mask 52 ms, both growing with the depth. only where Vulkan takes the sparse path anyway (KV at least
    // 2 x n_top_k and 4096), the dense attention over the mask is faster below (13.10: dense 821.7 against 878.2 t/s
    // at depth 4096, equal at 2048). 13.10 on the 395: PP at 122880 810.4 against 781.8 t/s, the attention 810 against
    // 909 ms, the mask fill 52 -> 3 ms, TG 25.74 against 25.39, KLD 0.015593 against 0.015741, FLASH_ATTN_LIST 7/7;
    // the compute buffer stays (its peak is in the indexer). LLAMA_QSA_FA_LIST: 1 = on (default), 0 = the mask
    static const int list_mode = [] {
        const char * env = std::getenv("LLAMA_QSA_FA_LIST");
        return env ? std::atoi(env) : 1;
    }();
    // LLAMA_QSA_FA_LIST_MIN: the KV size from which, -1 (default) = the sparse threshold of Vulkan above
    static const int64_t list_min = [] {
        const char * env = std::getenv("LLAMA_QSA_FA_LIST_MIN");
        return env ? (int64_t) std::atoll(env) : (int64_t) -1;
    }();
    bool use_list = false;
    if (list_mode == 1 && from_pos && cparams.flash_attn) {
        ggml_backend_dev_t dev_l = model.dev_layer(il);
        const bool vk_l = std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev_l)), "Vulkan") == 0;
        use_list = vk_l && n_kv_m >= (list_min >= 0 ? list_min : std::max<int64_t>(4096, 2*top_k->ne[0]));
    }

    ggml_tensor * kq_mask_all = nullptr;
    ggml_tensor * kept        = nullptr;

    // the values go into the graph before the mask they are scattered into (see the expand below): the -inf mask is
    // the largest tensor of the layer and should be the last thing allocated before the attention
    if (from_pos) {
        if (!qsa_pos_inp) {
            auto pos_inp = std::make_unique<llm_graph_input_qsa_pos>(mctx_hyb);

            pos_inp->cell_pos = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_kv_m, 1, n_str_m);
            pos_inp->tok_pos  = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, 1, n_tps_m, n_str_m);
            pos_inp->neg_inf  = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, 1, 1, 1, 1);

            ggml_set_input(pos_inp->cell_pos);
            ggml_set_input(pos_inp->tok_pos);
            ggml_set_input(pos_inp->neg_inf);

            if (pos_multi) {
                pos_inp->tok_half = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, 1, n_tps_m, n_str_m);
                ggml_set_input(pos_inp->tok_half);
            }

            qsa_pos_inp = pos_inp.get();
            res->add_input(std::move(pos_inp));
        }

        // positions of the selected cells: [1, n_top_k*n_batch, 1, n_stream] -> [1, n_top_k, n_batch, n_stream]
        ggml_tensor * idx = ggml_is_contiguous(top_k) ? top_k : ggml_cont(ctx0, top_k);
        idx = ggml_reshape_3d(ctx0, idx, top_k->ne[0]*top_k->ne[1], 1, top_k->ne[3]);

        ggml_tensor * cell_pos = ggml_get_rows(ctx0, qsa_pos_inp->cell_pos, idx);
        cell_pos = ggml_reshape_4d(ctx0, cell_pos, 1, top_k->ne[0], top_k->ne[1], top_k->ne[3]);

        // 0 where the cell is not after the token, -inf where it is (or holds another sequence, or nothing):
        // log(1 - step(cell - token)). with several sequences in the stream: log(1 - step(|cell - mid| - half)), 0 inside
        // the range of the token's sequence up to the token, -inf outside it
        if (qsa_pos_inp->tok_half) {
            kept = ggml_sub(ctx0, cell_pos, qsa_pos_inp->tok_pos);
            kept = ggml_abs(ctx0, kept);
            kept = ggml_sub(ctx0, kept, qsa_pos_inp->tok_half);
        } else {
            kept = ggml_sub(ctx0, cell_pos, qsa_pos_inp->tok_pos);
        }
        kept = ggml_step(ctx0, kept);
        kept = ggml_scale_bias(ctx0, kept, -1.0f, 1.0f);
        kept = ggml_log(ctx0, kept);
        ggml_build_forward_expand(gf, kept);

        // a node, so that the allocator frees it after the attention as it did the fill of the mask view
        if (!use_list) {
            kq_mask_all = ggml_repeat_4d(ctx0, qsa_pos_inp->neg_inf, 1, n_kv_m, n_tps_m, n_str_m);
        }
    } else {
        // view of the mask with rows of size 1, to read the selected rows out of:
        // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream].
        // built once for all QSA layers: the mask sits on the CPU, so a view per layer makes the
        // scheduler copy the whole mask to the backend per layer, n_kv*n_batch*2 bytes each time
        if (!qsa_kq_mask_rows) {
            qsa_kq_mask_rows = ggml_view_4d(ctx0, kq_mask, 1, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[3], kq_mask->nb[0], kq_mask->nb[1], kq_mask->nb[2], 0);
        }

        ggml_tensor * kq_mask_rows = qsa_kq_mask_rows;

        // gather the mask value of each selected cell: [1, n_top_k, n_batch, n_stream]
        // scattering these back beats unmasking with zeros and adding the whole mask afterwards,
        // which reads and writes n_kv values per token where only n_top_k of them can change
        kept = ggml_get_rows(ctx0, kq_mask_rows, top_k_3d);
        ggml_build_forward_expand(gf, kept);

        // prepare new kq mask - starts filled with -INFINITY. fill the row view and not the mask, so
        // that both readers take the same copy
        kq_mask_all = ggml_fill(ctx0, kq_mask_rows, -INFINITY);
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    if (use_list) {
        // the values f16 as the attention reads them, [1, n_top_k, n_batch, n_stream] -> [n_top_k, n_batch, 1, n_stream]
        ggml_tensor * vals = ggml_cast(ctx0, kept, GGML_TYPE_F16);
        vals = ggml_reshape_4d(ctx0, vals, top_k->ne[0], top_k->ne[1], 1, top_k->ne[3]);
        ggml_tensor * idx = ggml_is_contiguous(top_k_3d) ? top_k_3d : ggml_cont(ctx0, top_k_3d);

        // the attention as build_attn_mha does it for ggml_flash_attn_ext
        const bool v_trans  = v->nb[1] > v->nb[2];
        const auto n_stream = k->ne[3];
        q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[2]*(q->ne[2]/n_stream), 0);
        q = ggml_permute(ctx0, q, 0, 2, 1, 3);
        k = ggml_permute(ctx0, k, 0, 2, 1, 3);
        v = ggml_permute(ctx0, v, 0, 2, 1, 3);
        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }
        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }
        ggml_tensor * cur = ggml_flash_attn_list(ctx0, q, k, v, vals, idx, kq_scale);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        cb(cur, "kqv_out", il);

        if (inp->self_v_rot) {
            cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
        }
        return cur;
    }

    // fork: without the list attention (CUDA, CPU) the -inf mask of the selected cells is n_kv x n_batch f16, 3.3 GiB at
    // 214K x 8192, and --phase-mem shrank the prompt ubatch with the depth to fit it (the 3090, 15.10: 8192 at 4K, 4096
    // at 158K, 3072 at 214K), so a turn of ~3.7K tokens took two ubatches, each uploading all the experts in RAM (PP of
    // the turns 762 at 0-20K, 530 at 160-200K, 432 past 200K). LLAMA_QSA_MASK_MIB=<MiB> (0, off by default until the
    // 3090 measures it): the queries go in chunks whose mask fits, one chunk's mask alive at a time; each query attends
    // as before (CPU: the same attention output, Vulkan with LLAMA_QSA_FA_LIST=0: equal to float rounding)
    static const int64_t mask_mib = [] {
        const char * env = std::getenv("LLAMA_QSA_MASK_MIB");
        return env ? (int64_t) std::atoll(env) : (int64_t) 0;
    }();
    // LLAMA_QSA_MASK_CHUNK=<tokens>: the chunk itself (tests)
    static const int64_t mask_chunk = [] {
        const char * env = std::getenv("LLAMA_QSA_MASK_CHUNK");
        return env ? (int64_t) std::atoll(env) : (int64_t) 0;
    }();
    const int64_t n_tok_q = top_k->ne[1];
    const int64_t chunk   = mask_chunk > 0 ? mask_chunk :
        mask_mib > 0 ? std::max<int64_t>(256, (mask_mib*1024*1024 / (2*n_kv_m)) / 256 * 256) : 0;
    if (from_pos && n_str_m == 1 && chunk > 0 && n_tok_q > chunk) {
        std::vector<ggml_tensor *> parts;
        for (int64_t t0 = 0; t0 < n_tok_q; t0 += chunk) {
            const int64_t nc = std::min(chunk, n_tok_q - t0);
            ggml_tensor * kept_c = ggml_view_4d(ctx0, kept, 1, kept->ne[1], nc, 1, kept->nb[1], kept->nb[2], kept->nb[3], t0*kept->nb[2]);
            ggml_tensor * idx_c  = ggml_view_3d(ctx0, top_k_3d, top_k_3d->ne[0], nc, 1, top_k_3d->nb[1], top_k_3d->nb[2], t0*top_k_3d->nb[1]);
            ggml_tensor * mask_c = ggml_repeat_4d(ctx0, qsa_pos_inp->neg_inf, 1, n_kv_m, nc, 1);
            mask_c = ggml_set_rows(ctx0, mask_c, kept_c, idx_c);
            mask_c = ggml_view_4d(ctx0, mask_c, n_kv_m, nc, 1, 1, mask_c->nb[2], mask_c->nb[3], mask_c->nb[3], 0);
            ggml_tensor * q_c = ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], nc, q->nb[1], q->nb[2], t0*q->nb[2]);
            ggml_tensor * cur_c = build_attn_mha(q_c, k, v, nullptr, mask_c, nullptr, nullptr, top_k->ne[0], kq_scale, il);
            // in order: the mask of the next chunk can take the memory of this one
            ggml_build_forward_expand(gf, cur_c);
            parts.push_back(cur_c);
        }
        // pairwise, so that each row is copied log2(chunks) times
        while (parts.size() > 1) {
            std::vector<ggml_tensor *> next;
            for (size_t i = 0; i + 1 < parts.size(); i += 2) {
                next.push_back(ggml_concat(ctx0, parts[i], parts[i + 1], 1));
            }
            if (parts.size() % 2) {
                next.push_back(parts.back());
            }
            parts.swap(next);
        }
        ggml_tensor * cur = parts[0];
        cb(cur, "kqv_out", il);
        if (inp->self_v_rot) {
            cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
        }
        return cur;
    }

    // modify KQ mask by restoring the elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, kept, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // kq_mask_all starts at -inf and only the top_k cells get their mask value back, so a row can
    // never hold more than top_k->ne[0] finite entries - the bound ggml_flash_attn_ext_set_n_kv_max wants.
    // Backends without a sparse path ignore it and run the same dense masked attention.
    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp_fork::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_fork_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il,
        bool                      keys_only) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    if (keys_only) {
        // K and V of every row to the cache, the indexer keys too, no query: a later decode attends over them
        GGML_ASSERT(qsa);
        build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il, true);

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        cb(Vcur, "Vcur", il);

        // as build_attn_qsa stores them
        if (inp->self_k_rot) {
            Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp->self_k_rot);
        }
        if (inp->self_v_rot) {
            Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp->self_v_rot);
        }
        ggml_tensor * k_store = inp->mctx->cpy_k(ctx0, Kcur, inp->get_k_idxs(), il);
        ggml_tensor * v_store = inp->mctx->cpy_v(ctx0, Vcur, inp->get_v_idxs(), il);
        if (mctx_hyb->is_kv_host()) {
            ggml_backend_sched_set_tensor_backend(sched, k_store, backend_cpu);
            ggml_backend_sched_set_tensor_backend(sched, v_store, backend_cpu);
        }
        ggml_build_forward_expand(gf, k_store);
        ggml_build_forward_expand(gf, v_store);
        return nullptr;
    }

    ggml_tensor * top_k = qsa ? build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il) : nullptr;

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    // the sigmoid reads the gate in place from the q projection (no copy of n_embd_head * n_head floats per token) and
    // goes into the graph here, before the attention: the projection (twice the gate) can then be freed after the q
    // norm instead of living through the attention (11.10: the 395 compute buffer 7905 -> 8121 MiB with the sigmoid at
    // the end). the MUL below takes the attention output in the same [n_embd_head, n_head, n_tokens] shape
    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    cb(gate, "gate_reshaped", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);
    ggml_build_forward_expand(gf, gate_sigmoid);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_k) {
        cur = build_attn_qsa(inp, mctx_hyb, Qcur, Kcur, Vcur, top_k, kq_scale, mctx_hyb->is_kv_host(), il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    if (ggml_is_contiguous(cur)) {
        cur = ggml_mul(ctx0, ggml_reshape_3d(ctx0, cur, n_embd_head, n_head, n_tokens), gate_sigmoid);
        cur = ggml_reshape_2d(ctx0, cur, n_embd_head * n_head, n_tokens);
    } else {
        cur = ggml_mul(ctx0, cur, ggml_reshape_2d(ctx0, gate_sigmoid, n_embd_head * n_head, n_tokens));
    }
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp_fork::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    // deferred rollback (llm_graph_input_rs::lazy_mode): a short batch runs the net that records instead of writing
    // snapshots and reads its states from their rows itself; any other first replays the records the states still
    // carry, in place, so the gather and the plain net see the states they expect
    const int lazy_mode = inp->lazy_mode;

    ggml_tensor * state = nullptr;
    if (lazy_mode != 1) {
        ggml_tensor * ssm_src = ssm_states_all;
        if (lazy_mode == 2) {
            ssm_src = ggml_gated_delta_net_replay(ctx0, ssm_states_all, mctx_cur->get_rec_l(il), inp->lazy_info,
                    (int) head_v_dim, (int) num_v_heads, (int) num_k_heads);
        }
        state = build_rs(inp, ssm_src, hparams.n_embd_s(), n_seqs);
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
        cb(state, "state_predelta", il);
    }

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN and the net that records
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch) && lazy_mode != 1) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = nullptr;
    if (lazy_mode == 1) {
        output = ggml_gated_delta_net_lazy(ctx0, q_conv, k_conv, v_conv, gate, beta, ssm_states_all,
                mctx_cur->get_rec_l(il), inp->lazy_info);
        output = ggml_reshape_4d(ctx0, output, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
        cb(output, "attn_output", il);
    } else {
        output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);
    }

    // the gate of the output norm goes into the graph here, after the GDN and before the norm: its matmul and reshape
    // would otherwise land between the norm and the sigmoid (Vulkan runs RMS_NORM, MUL, SIGMOID, MUL as one shader when
    // they are adjacent), and through the GDN only the layer input stays alive (n_embd floats a token, not d_inner)
    ggml_build_forward_expand(gf, z_2d);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp_fork::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple_fork : public llm_graph_input_i {
public:
    llm_graph_input_ple_fork(const llama_model_qwen4exp_fork & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple_fork() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_fork_context *>(params.mctx)->get_attn();
        const int64_t n_tokens = params.ubatch.n_tokens;
        return rows ? rows->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * n_tokens
                    : emb->ne[1] == n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens], table in memory
    ggml_tensor * emb  = nullptr;   // F32 [ple_head_dim * ple_n_heads, n_tokens], table on disk

    const llama_model_qwen4exp_fork & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
    std::vector<float>       gathered;
};

// the PLE table rows of a ubatch: n_heads per token, from the token and its n_gram - 1 predecessors (prev, oldest
// first, LLAMA_TOKEN_NULL where missing)
static void qwen4exp_ple_rows(const llama_hparams & hp, const llama_ubatch & ub, const std::vector<llama_token> & prev,
                              std::vector<int32_t> & idx) {
    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    // a mixed ubatch has token ids and embd rows, its embd rows hold a placeholder token (upstream 0bb496dbd)
    auto tok_of = [&](int64_t k) -> llama_token {
        const bool is_embd = !ub.token || (ub.is_mixed() && ub.type[k]);
        return is_embd ? img_tok : ub.token[k];
    };

    const int64_t n_tokens = ub.n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    idx.assign(n_heads * n_tokens, 0);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }
}

void llm_graph_input_ple_fork::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_prev   = hp.ple_ngram_size - 1;

    std::vector<int32_t> idx;

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    qwen4exp_ple_rows(hp, *ubatch, prev, idx);

    if (rows) {
        // a lazily mapped table: ask the OS for the rows before the gather reads them (upstream 185103dcf). the
        // row cache of -lzm dio reads the file itself, the tensor does not exist then
        if (pmodel.can_prefetch.count(pmodel.per_layer_tok_embd)) {
            llama_prefetch_rows(pmodel.per_layer_tok_embd, idx.data(), idx.size());
        }
        ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
        return;
    }

    // the table was never loaded: gather the rows here, the graph gets them ready to use
    GGML_ASSERT(pmodel.ple_cache && "PLE row cache is missing");

    // gathered while the ubatch before ran, when the row list matches (see ple_prefetch_start)
    if (!pmodel.ple_prefetch_take(idx, gathered)) {
        gathered.resize(idx.size()*hp.ple_head_dim);
        pmodel.ple_cache->get_rows(idx.data(), (int64_t) idx.size(), gathered.data());
    }

    ggml_backend_tensor_set(emb, gathered.data(), 0, ggml_nbytes(emb));
}

// the PLE rows of a ubatch that is not stored yet: the predecessors before its start from the cells (get_prev, as
// llama_kv_cache::get_prev_tokens), the ones inside it from its own tokens. 0 when done, else why not: 1 not tokens
// only (an image or a mixed ubatch orders its predecessors by position runs this cannot see), 2 a token in several
// sequences, 3 a gap in its positions
template <typename F>
static int qwen4exp_ple_rows_ahead(const llama_hparams & hp, const llama_ubatch & next, F && get_prev,
                                   std::vector<int32_t> & idx) {
    if (next.token == nullptr || next.embd != nullptr || next.is_mixed()) {
        return 1;
    }

    const int64_t n_tokens = next.n_tokens;
    const int64_t n_prev   = hp.ple_ngram_size - 1;

    std::vector<llama_token> prev;
    get_prev(next, (uint32_t) n_prev, prev);

    std::unordered_map<llama_seq_id, std::unordered_map<llama_pos, llama_token>> own;
    std::unordered_map<llama_seq_id, llama_pos> pos_min;
    for (int64_t i = 0; i < n_tokens; ++i) {
        if (next.n_seq_id[i] != 1) {
            return 2;
        }
        const llama_seq_id s = next.seq_id[i][0];
        own[s][next.pos[i]] = next.token[i];
        auto it = pos_min.find(s);
        if (it == pos_min.end() || next.pos[i] < it->second) {
            pos_min[s] = next.pos[i];
        }
    }
    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = next.seq_id[i][0];
        for (int64_t j = 0; j < n_prev; ++j) {
            const llama_pos p = next.pos[i] - (llama_pos) (n_prev - j);
            if (p < pos_min[s]) {
                continue;
            }
            const auto it = own[s].find(p);
            if (it == own[s].end()) {
                return 3;
            }
            prev[i*n_prev + j] = it->second;
        }
    }

    qwen4exp_ple_rows(hp, next, prev, idx);
    return 0;
}

void llama_model_qwen4exp_fork::ple_prefetch_start(const llama_ubatch & next, const llama_kv_cache_context * mctx) const {
    static const bool enabled = [] {
        const char * env = std::getenv("LLAMA_PLE_PREFETCH");
        return env == nullptr || std::atoi(env) != 0;
    }();
    // why a prefetch did not start, once per reason: on the 395 (12.10 night, -b 16384 -ub 4096) the take never logged
    auto skip = [](int reason, const char * what) {
        static std::atomic<uint32_t> seen { 0 };
        if (!(seen.fetch_or(1u << reason) & (1u << reason))) {
            LLAMA_LOG_INFO("ple_prefetch_start: no prefetch: %s\n", what);
        }
    };
    if (!enabled) {
        return;
    }
    if (!ple_cached || !ple_cache) {
        skip(0, "the PLE table is loaded, the graph gathers its rows");
        return;
    }
    if (mctx == nullptr) {
        skip(1, "the next ubatch is not one of tokens only");
        return;
    }

    std::vector<int32_t> idx;
    const int why = qwen4exp_ple_rows_ahead(hparams, next,
        [mctx](const llama_ubatch & ub, uint32_t n, std::vector<llama_token> & res) { mctx->get_prev_tokens(ub, n, res); }, idx);
    if (why != 0) {
        static const char * what[] = { "", "the next ubatch is not one of tokens only",
            "a token of the next ubatch is in several sequences", "a gap in the positions of the next ubatch" };
        skip(why, what[why]);
        return;
    }

    if (!ple_pf) {
        ple_pf = std::make_unique<ple_prefetch>();
    }
    ple_prefetch & pf = *ple_pf;

    std::lock_guard<std::mutex> lock(pf.mutex);
    pf.finish();
    pf.idx = std::move(idx);
    pf.rows.resize(pf.idx.size()*hparams.ple_head_dim);
    pf.active = true;
    pf.n_started++;
    pf.worker = std::thread([this, &pf] {
        ple_cache->get_rows(pf.idx.data(), (int64_t) pf.idx.size(), pf.rows.data());
    });
}

void llama_model_qwen4exp_fork::ple_warm(const llama_ubatch & next, const llama_kv_cache & kv, int64_t n_new) const {
    static const bool enabled = [] {
        const char * env = std::getenv("LLAMA_PLE_WARM");
        return env == nullptr || std::atoi(env) != 0;
    }();
    if (!enabled || !ple_cached || !ple_cache || n_new <= 0) {
        return;
    }

    std::vector<int32_t> idx;
    if (qwen4exp_ple_rows_ahead(hparams, next,
            [&kv](const llama_ubatch & ub, uint32_t n, std::vector<llama_token> & res) { kv.get_prev_tokens(ub, n, res); }, idx) != 0) {
        return;
    }
    // the last n_new tokens: the ones before were read by an earlier call
    const int64_t n_heads = hparams.ple_n_heads;
    const int64_t n_skip  = std::max<int64_t>(0, (int64_t) next.n_tokens - n_new);
    idx.erase(idx.begin(), idx.begin() + n_skip*n_heads);

    if (!ple_wm) {
        ple_wm = std::make_unique<ple_warmer>();
        ple_warmer * wm = ple_wm.get();
        // one worker for the life of the model; a read of a few rows is a few page faults or block reads
        wm->worker = std::thread([this, wm] {
            std::vector<float> scratch;
            std::unique_lock<std::mutex> lock(wm->mutex);
            while (true) {
                wm->cv.wait(lock, [wm] { return wm->stop || !wm->queue.empty(); });
                if (wm->stop) {
                    return;
                }
                std::vector<int32_t> rows = std::move(wm->queue.front());
                wm->queue.erase(wm->queue.begin());
                lock.unlock();
                scratch.resize(rows.size()*hparams.ple_head_dim);
                ple_cache->get_rows(rows.data(), (int64_t) rows.size(), scratch.data());
                lock.lock();
                wm->n_rows += (int64_t) rows.size();
            }
        });
    }
    {
        std::lock_guard<std::mutex> lock(ple_wm->mutex);
        ple_wm->queue.push_back(std::move(idx));
    }
    ple_wm->cv.notify_one();
}

bool llama_model_qwen4exp_fork::ple_prefetch_take(const std::vector<int32_t> & idx, std::vector<float> & dst) const {
    if (!ple_pf) {
        return false;
    }
    ple_prefetch & pf = *ple_pf;

    std::lock_guard<std::mutex> lock(pf.mutex);
    if (!pf.active) {
        return false;
    }
    if (pf.idx != idx) {
        static std::atomic<bool> logged { false };
        if (!logged.exchange(true)) {
            LLAMA_LOG_INFO("%s: the PLE rows gathered ahead are not the ones of this ubatch (%zu against %zu rows)\n",
                    __func__, pf.idx.size(), idx.size());
        }
        return false;
    }
    pf.finish();
    pf.active = false;
    if (pf.n_used++ == 0) {
        // the server kills its process at the end, so the count at unload rarely makes it to the log
        LLAMA_LOG_INFO("%s: the PLE rows of a ubatch were gathered while the one before computed\n", __func__);
    }
    std::swap(dst, pf.rows);
    return true;
}

// the fork's step before the graph of a ubatch runs (llama_context::process_ubatch): the PLE rows of the next ubatch,
// or for the last ubatch of a decode those of hint, the first ubatch of the next decode (llama_set_next_tokens).
// true when the hint was taken
bool llama_fork_before_compute(const llama_model & model, llama_memory_context_i * mctx, const llama_ubatch * hint) {
    const auto * hyb = dynamic_cast<const llama_memory_hybrid_idx_fork_context *>(mctx);
    if (hyb == nullptr) {
        return false;
    }
    const auto * qm = dynamic_cast<const llama_model_qwen4exp_fork *>(&model);
    if (qm == nullptr || !qm->ple_cached) {
        return false;
    }
    const llama_ubatch * next = hyb->peek_next();
    if (next != nullptr) {
        qm->ple_prefetch_start(*next, hyb->get_attn());
        return false;
    }
    if (hint != nullptr) {
        qm->ple_prefetch_start(*hint, hyb->get_attn());
        return true;
    }
    return false;
}

bool llama_fork_gdn_lazy(const llama_model & model) {
    return dynamic_cast<const llama_model_qwen4exp_fork *>(&model) != nullptr;
}

void llama_fork_prefetch_tokens(const llama_model & model, llama_memory_i * mem, const llama_ubatch & next, int64_t n_new) {
    const auto * qm  = dynamic_cast<const llama_model_qwen4exp_fork *>(&model);
    const auto * hyb = dynamic_cast<const llama_memory_hybrid_idx_fork *>(mem);
    if (qm == nullptr || hyb == nullptr || hyb->get_mem_attn() == nullptr) {
        return;
    }
    qm->ple_warm(next, *hyb->get_mem_attn(), n_new);
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp_fork::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    // when every slot ends inside x, the tails are read from x: the conv is then the only reader of the concat, which
    // Vulkan runs inside the conv (CONCAT_SSM_CONV, the conv reads the state and x where they are)
    const int64_t n_seq_tokens = x->ne[1];
    const bool tail_from_x = n_seq_tokens >= state_cols + n_slots - 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = nullptr;
        if (tail_from_x) {
            // columns s_idx - state_cols .. s_idx - 1 of x, the same as columns s_idx .. of the concat
            tail = ggml_view_3d(ctx0, x, channels, state_cols, n_seqs, x->nb[1], x->nb[2],
                    (s_idx - state_cols) * x->nb[1]);
            tail = ggml_transpose(ctx0, tail);
        } else {
            tail = ggml_view_3d(ctx0, conv_input,
                    state_cols, channels, n_seqs,
                    conv_input->nb[1], conv_input->nb[2],
                    ggml_row_size(conv_input->type, s_idx));
        }

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp_fork::graph::build_inp_ple(
        const llama_memory_hybrid_idx_fork_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple_fork>(
            static_cast<const llama_model_qwen4exp_fork &>(model), mctx_hyb->get_attn());

    ggml_tensor * emb = nullptr;

    if (static_cast<const llama_model_qwen4exp_fork &>(model).ple_cached) {
        // set_input reads the rows off disk, so the graph only sees the result
        ple_inp->emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.ple_head_dim * n_heads, n_tokens);
        ggml_set_input(ple_inp->emb);
        emb = ple_inp->emb;
    } else {
        ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
        ggml_set_input(ple_inp->rows);

        // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
        emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, ple_inp->rows);
        emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    }

    res->add_input(std::move(ple_inp));
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp_fork::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
