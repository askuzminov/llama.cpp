#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <vector>

class llm_graph_result;

struct llama_model;

// enabled with LLAMA_MOE_CACHE_STATS=<report period in decode steps>, 1 means the default period
bool llama_moe_stats_enabled();

// measures what a GPU-resident LRU cache of MoE experts would give, without building one.
// replays the real routing ids of every ubatch through LRU simulations of several cache
// sizes and insert budgets, and reports the decode hit rate, the upload traffic each
// combination needs and the host reads it removes.
class llama_moe_stats {
public:
    llama_moe_stats(const llama_model & model);

    // read back the routing ids recorded by the graph and feed the simulations
    void add_ubatch(const llm_graph_result * res, ggml_backend_sched_t sched, uint32_t n_tokens);

    void report() const;

private:
    struct sim_layer {
        std::vector<int32_t> prev;
        std::vector<int32_t> next;
        std::vector<uint8_t> cached;

        int32_t head     = -1;
        int32_t tail     = -1;
        int32_t n_cached = 0;
        int32_t n_ins    = 0; // inserts done in the current step
    };

    struct sim {
        int32_t n_slots = 0;
        int32_t max_ins = 0; // max inserts per layer per step

        std::vector<sim_layer> layers;

        uint64_t n_acc = 0; // decode accesses
        uint64_t n_hit = 0; // decode hits
        uint64_t b_hit = 0; // expert bytes served from the cache on decode steps
        uint64_t b_ins = 0; // expert bytes uploaded on decode steps
    };

    struct moe_layer {
        int      il        = -1;
        bool     on_host   = false;
        uint64_t bytes     = 0; // up + gate + down of one expert
        uint64_t n_acc     = 0; // decode accesses
        std::vector<uint32_t> freq;
    };

    int  layer_idx(int il);
    void access(int idx, int32_t e, bool is_dec);

    const llama_model & model;

    int64_t n_expert      = 0;
    int64_t n_expert_used = 0;

    uint32_t period       = 0;
    uint32_t n_tokens_dec = 0; // a step with at most this many tokens counts as decode

    uint64_t n_steps_dec  = 0;
    uint64_t n_steps_all  = 0;
    uint64_t n_tok_dec    = 0;

    std::vector<int>       il2idx;
    std::vector<moe_layer> mlayers;
    std::vector<sim>       sims;

    std::vector<int32_t>       buf;
    std::vector<ggml_backend_t> backends;
};
