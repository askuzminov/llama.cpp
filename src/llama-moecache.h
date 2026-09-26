#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

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

// device cache of the MoE experts that stay in host memory (--moe-cache)
// the graph computes the cached experts from device slots and the host mul_mat_id nodes skip them.
// one context owns one cache, the slots are uploaded on a separate backend instance of the device.
class llama_moe_cache {
public:
    // the graph uses the cache only for batches of at most this many tokens
    static constexpr int64_t N_TOKENS_MAX = 128;

    struct layer {
        int il = -1;

        bool merged = false; // w_up holds gate and up

        // host expert weights
        ggml_tensor * w_up   = nullptr;
        ggml_tensor * w_gate = nullptr;
        ggml_tensor * w_down = nullptr;

        // device slots [ne0, ne1, n_slots + 1], the last slot stays zero for the experts that are not cached
        ggml_tensor * s_up   = nullptr;
        ggml_tensor * s_gate = nullptr;
        ggml_tensor * s_down = nullptr;

        ggml_tensor * table = nullptr; // device I32 [1, n_expert]: slot of each expert, n_slots when not cached
        ggml_tensor * skip  = nullptr; // host I32 [n_expert]: 1 when the expert is cached
        ggml_tensor * ids   = nullptr; // host I32 [n_used_max*N_TOKENS_MAX]: routing ids of the last step
    };

    llama_moe_cache(const llama_model & model, int32_t n_slots_req);
    ~llama_moe_cache();

    // select the layers and allocate the slots from the free device memory, called once
    void init(ggml_backend_sched_t sched);

    bool is_init() const { return initialized; }
    bool ready()   const { return n_slots > 0; }

    // nullptr when the experts of the layer are not cached
    const layer * get_layer(int il) const;

    ggml_backend_dev_t get_dev() const { return dev; }

    // update the LRU from the routing ids of the graph and start the uploads, call after the graph is computed
    // the uploads run during the next graph and the graph after it can use them
    void update(const llm_graph_result * res);

    void report() const;

private:
    struct lru {
        std::vector<int32_t> prev;   // [n_expert]
        std::vector<int32_t> next;   // [n_expert]
        std::vector<int32_t> slot;   // [n_expert] slot of the expert, -1 when not cached
        std::vector<int32_t> expert; // [n_slots] expert in the slot
        std::vector<uint8_t> live;   // [n_slots] 1 when the graph can read the slot

        std::vector<int32_t> uploads_new;  // slots with an upload that the last update() started
        std::vector<int32_t> uploads_done; // slots with an upload that finished, but is not published yet
        std::vector<int32_t> evicted;      // published experts that lost their slot

        int32_t head   = -1;
        int32_t tail   = -1;
        int32_t n_fill = 0;
    };

    void lru_unlink(lru & c, int32_t e);
    void lru_push  (lru & c, int32_t e);
    void insert    (size_t idx, int32_t e);

    const llama_model & model;

    const int32_t n_slots_req;

    bool initialized = false;

    int64_t n_expert = 0;
    int32_t n_slots  = 0;
    size_t  b_slot   = 0; // bytes of one slot on every cached layer

    ggml_backend_dev_t dev        = nullptr;
    ggml_backend_t     backend    = nullptr; // compute backend of dev in the scheduler, not owned
    ggml_backend_t     backend_up = nullptr; // separate instance for the uploads

    // starts the queued uploads on backends that wait for a submit, nullptr when the device has no events
    ggml_backend_event_t ev_up = nullptr;

    bool ev_up_recorded = false;

    ggml_context_ptr        ctx_dev;
    ggml_context_ptr        ctx_host;
    ggml_backend_buffer_ptr buf_dev;
    ggml_backend_buffer_ptr buf_host;

    ggml_tensor * table_all = nullptr; // device I32 [n_expert, n_layers]

    std::vector<int32_t> table_host; // host copy of table_all
    std::vector<int>     il2idx;
    std::vector<layer>   layers;
    std::vector<lru>     lrus;

    uint64_t n_steps   = 0;
    uint64_t n_acc     = 0; // routed experts seen by update()
    uint64_t n_hit     = 0; // of them computed from the slots
    uint64_t n_ins     = 0;
    uint64_t b_ins     = 0; // uploaded bytes
    int64_t  t_wait_us = 0; // host time spent waiting for the uploads
};
