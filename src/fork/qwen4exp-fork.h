#pragma once

// the fork's qwen4exp: QSA over pooled block keys kept across passes (llama-memory-hybrid-idx-fork), the MTP draft
// head, PLE rows read on demand (llama_row_cache). models.h keeps the upstream struct; llama_model_create picks this
// one unless LLAMA_UPSTREAM lists qwen4exp (see llama-fork.h)

#include "models/models.h"
#include "llama-row-cache-fork.h"

class llama_memory_hybrid_idx_fork_context;

struct llama_model_qwen4exp_fork : public llama_model_base {
    llama_model_qwen4exp_fork(const struct llama_model_params & params) : llama_model_base(params) {}
    ~llama_model_qwen4exp_fork() override;

    class llm_graph_input_qsa;

    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    // under -lzm dio the PLE table is never loaded: per_layer_tok_embd stays null and the rows
    // are read from the file on demand instead
    bool                             ple_cached = false;
    std::unique_ptr<llama_row_cache> ple_cache;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);
    protected:
        // tag-dispatched ctor for graph_mtp: binds the members without building the trunk
        struct no_build_t {};
        graph(const llama_model & model, const llm_graph_params & params, no_build_t) :
            llm_build_delta_net_base(params), model(model) {}

        // HC replaces every layer norm: residual is [n_embd, hc, n_tokens]
        ggml_tensor * build_hc_mix(
                    ggml_tensor * x,
                    ggml_tensor * w_norm,
                    ggml_tensor * w_down,
                    ggml_tensor * w_up,
                    ggml_tensor * w_inject,
                    ggml_tensor ** inject,
                            int   il);

        ggml_tensor * build_hc_combine(
                    ggml_tensor * residual,
                    ggml_tensor * block_out,
                    ggml_tensor * inject,
                            int   il);

        ggml_tensor * build_layer_attn(
              llm_graph_input_attn_kv * inp_attn,
  const llama_memory_hybrid_idx_fork_context * mctx_hyb,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        // dense self-attention restricted to the cells that top_k names
        ggml_tensor * build_attn_qsa(
        llm_graph_input_attn_kv * inp,
                    ggml_tensor * q_cur,
                    ggml_tensor * k_cur,
                    ggml_tensor * v_cur,
                    ggml_tensor * top_k,
                          float   kq_scale,
                           bool   kv_host,
                            int   il);

        // the QSA cache layout inputs do not depend on the layer, only on its compress ratio,
        // so the layers sharing a ratio share one input set
        std::map<uint32_t, llm_graph_input_qsa *> qsa_inps;

        // row view of the KQ mask input, shared by the QSA layers - see build_attn_qsa
        ggml_tensor * qsa_kq_mask_rows = nullptr;

        // QSA: token indices this layer's queries may attend to, or nullptr for dense
        ggml_tensor * build_qsa_top_k(
  const llama_memory_hybrid_idx_fork_context * mctx_hyb,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                    ggml_tensor * kq_mask,
                            int * sections,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // build_rs writes the state tensor in place, so one gather per cache tensor is reused
        std::map<ggml_tensor *, ggml_tensor *> rs_rows;

        // one conv history per cache tensor: delta-net and PLE each have their own
        ggml_tensor * build_conv_state_at(
             llm_graph_input_rs * inp,
                    ggml_tensor * conv_states_all,
                    ggml_tensor * x,
                        int64_t   state_cols,
                        int64_t   channels,
                            int   il);

        ggml_tensor * build_inp_ple(
  const llama_memory_hybrid_idx_fork_context * mctx_hyb);

        ggml_tensor * build_ple(
             llm_graph_input_rs * inp,
                    ggml_tensor * emb,
                    ggml_tensor * hidden,
                            int   il);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const llama_model & model;
    };

    // LLM_GRAPH_TYPE_DECODER_MTP draft head: one HC-wrapped dense-attention + MoE block
    struct graph_mtp : public graph {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
