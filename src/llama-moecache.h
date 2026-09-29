#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <vector>

class llm_graph_result;

struct llama_model;

// enabled with LLAMA_MOE_CACHE_STATS=<report period in decode steps>, 1 means the default period
bool llama_moe_stats_enabled();

// measures what a GPU-resident cache of MoE experts would give, without building one.
// records the real routing ids of every ubatch and, at each report, replays them through
// several cache policies, sizes and insert budgets. all of them see the same routing, so
// they compare without the noise of two runs. reports the decode hit rate, the upload
// traffic and the host reads of each combination.
class llama_moe_stats {
public:
    llama_moe_stats(const llama_model & model);

    // read back the routing ids recorded by the graph and add them to the trace
    void add_ubatch(const llm_graph_result * res, ggml_backend_sched_t sched, uint32_t n_tokens);

    // replay the trace and print the tables
    void report();

    // add the size and the upload limit of the real cache to the tables, call before the first report
    void set_cache(int32_t n_slots, int32_t max_ins);

private:
    enum policy {
        POLICY_LRU,  // --moe-cache
        POLICY_LRU1, // --moe-cache if a new expert could serve one step earlier
        POLICY_2Q,
        POLICY_LFU,  // LRU with frequency admission
        POLICY_OPT,  // Belady with the future of the replayed trace
        POLICY_COUNT,
    };

    // lists of an expert in a simulated layer
    enum list_id {
        LIST_NONE,
        LIST_MAIN, // LRU, 2Q Am
        LIST_IN,   // 2Q A1in
        LIST_OUT,  // 2Q A1out, no slot
        LIST_COUNT,
    };

    struct sim_layer {
        std::vector<int32_t>  prev;  // [n_expert]
        std::vector<int32_t>  next;  // [n_expert]
        std::vector<int32_t>  ready; // [n_expert] first step the expert can hit, -1 when it has no slot
        std::vector<int32_t>  gone;  // [n_expert] step in which a ready expert lost its slot
        std::vector<uint8_t>  list;  // [n_expert] list_id
        std::vector<uint16_t> freq;  // [n_expert] LFU counts
        std::vector<int32_t>  nu;    // [n_expert] OPT next use

        int32_t head[LIST_COUNT] = { -1, -1, -1, -1 };
        int32_t tail[LIST_COUNT] = { -1, -1, -1, -1 };
        int32_t size[LIST_COUNT] = {  0,  0,  0,  0 };

        int32_t n_ins  = 0; // inserts done in the current step
        int32_t n_seen = 0; // LFU accesses since the counts were halved

        void unlink(int32_t e);
        void push  (int li, int32_t e);
        void evict (int32_t e, int32_t step);
    };

    struct sim {
        policy  pol     = POLICY_LRU;
        int32_t n_slots = 0;
        int32_t max_ins = 0; // max inserts per layer per step
        int32_t delay   = 0; // steps from the miss to the first step the new expert can hit

        std::vector<sim_layer> layers;

        uint64_t n_acc  = 0; // decode accesses
        uint64_t n_hit  = 0; // decode hits
        uint64_t b_host = 0; // expert bytes read from host on decode steps, once per expert and step
        uint64_t b_ins  = 0; // expert bytes uploaded on decode steps
    };

    struct moe_layer {
        int      il      = -1;
        bool     on_host = false;
        uint64_t bytes   = 0; // up + gate + down of one expert
        uint64_t n_acc   = 0; // decode accesses
        uint64_t b_host  = 0; // expert bytes read from host on decode steps with no cache

        std::vector<uint32_t> freq;

        // routing of the steps since the last report, token-major
        std::vector<int32_t>  tr_ids;
        std::vector<uint32_t> tr_off;   // [n_steps + 1] start of each step in tr_ids
        std::vector<uint8_t>  tr_first; // 1 for the first access of an expert in its step
    };

    int  layer_idx(int il);
    std::vector<int32_t> table_slots() const;
    std::vector<int32_t> table_ins()   const;
    void sim_init(sim & s) const;
    void replay_layer    (sim & s, int idx) const;
    void replay_layer_opt(sim & s, int idx) const;
    void replay();

    const llama_model & model;

    int64_t n_expert      = 0;
    int64_t n_expert_used = 0;

    uint32_t period       = 0;
    uint32_t n_tokens_dec = 0; // a step with at most this many tokens counts as decode

    int32_t cache_slots = 0; // --moe-cache, 0 when there is no cache
    int32_t cache_ins   = 0;

    uint64_t n_steps_dec  = 0;
    uint64_t n_steps_all  = 0;
    uint64_t n_steps_rep  = 0; // decode steps at the last report
    uint64_t n_tok_dec    = 0;

    int64_t  t_replay_us  = 0; // time of the last replay

    std::vector<int>       il2idx;
    std::vector<moe_layer> mlayers;
    std::vector<sim>       sims;

    uint64_t tr_step0 = 0; // decode step at the start of the trace

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

    // select the layers and create the upload backend, called once before alloc()
    void init(ggml_backend_sched_t sched);

    // allocate the slots from the free device memory and upload the experts that the last release() kept
    void alloc();

    // free the slots, keep the cached experts of each layer in LRU order for the next alloc()
    void release();

    bool is_init() const { return initialized; }
    bool ready()   const { return n_slots > 0; }

    int32_t get_n_slots() const { return n_slots; }
    int32_t get_max_ins() const { return max_ins; }

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
        std::vector<uint64_t> seen;  // [n_expert] last step that counted the expert in b_host

        std::vector<int32_t> uploads_new;  // slots with an upload that the last update() started
        std::vector<int32_t> uploads_done; // slots with an upload that finished, but is not published yet
        std::vector<int32_t> evicted;      // published experts that lost their slot

        int32_t head   = -1;
        int32_t tail   = -1;
        int32_t n_fill = 0;
    };

    void   lru_unlink(lru & c, int32_t e);
    void   lru_push  (lru & c, int32_t e);
    void   insert    (size_t idx, int32_t e);
    size_t upload    (size_t idx, int32_t e, int32_t s);

    const llama_model & model;

    const int32_t n_slots_req;

    bool initialized  = false;
    bool alloc_logged = false; // the first alloc() logs at warn level

    int64_t n_expert  = 0;
    int32_t n_slots   = 0;
    int32_t n_slots_l = 0; // slots of the last alloc()
    int32_t max_ins   = 1; // max expert uploads per layer per step, 1 measured best up to 192 slots
    size_t  b_slot    = 0; // bytes of one slot on every cached layer
    int     n_moe     = 0; // MoE layers of the model

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

    std::vector<std::vector<int32_t>> hot; // [n_layers] experts cached at the last release(), most recent first

    uint64_t n_steps    = 0;
    uint64_t n_acc      = 0; // routed experts seen by update()
    uint64_t n_hit      = 0; // of them computed from the slots
    uint64_t n_ins      = 0;
    uint64_t b_ins      = 0; // uploaded bytes
    uint64_t b_host     = 0; // expert bytes the host read, once per expert and step
    int64_t  t_graph_us = 0; // host time spent waiting for the end of the graph
    int64_t  t_wait_us  = 0; // host time spent waiting for the uploads
    int64_t  t_upd_us   = 0; // host time of the rest of update(): LRU, upload calls, table

    uint64_t n_alloc    = 0; // alloc() calls that allocated slots
    uint64_t n_refill   = 0; // experts that alloc() uploaded again
    uint64_t b_refill   = 0;
    int64_t  t_alloc_us = 0; // host time of alloc() and release()
};
