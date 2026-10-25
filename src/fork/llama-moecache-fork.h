#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <functional>
#include <thread>
#include <unordered_map>
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
class llama_moe_stats_fork {
public:
    llama_moe_stats_fork(const llama_model & model);

    // read back the routing ids recorded by the graph and add them to the trace
    void add_ubatch(const llm_graph_result * res, ggml_backend_sched_t sched, uint32_t n_tokens);

    // replay the trace and print the tables
    void report();

    // add the size and the upload limit of the real cache to the tables, call before the first report
    void set_cache(int32_t n_slots, int32_t max_ins, bool lfu = false);

private:
    enum policy {
        POLICY_LRU,  // --moe-cache
        POLICY_LRU1, // --moe-cache if a new expert could serve one step earlier
        POLICY_2Q,
        POLICY_LFU,  // LRU with frequency admission
        POLICY_OPT,  // Belady with the future of the replayed trace
        POLICY_PLRU, // one pool of n_slots x layers slots for all layers, LRU over (layer, expert), as upstream's cache
        POLICY_PLFU, // the same pool with the frequency admission of POLICY_LFU
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

        // the routing of this layer that its router gives for the router input of the layer d before it (index d - 1),
        // n_pred_top ids per token, token-major, empty in a step without that layer
        std::vector<int32_t>  pr_ids[2];
        std::vector<uint32_t> pr_off[2]; // [n_steps + 1]

        // the prediction counts of this layer (replay_pred), by d and top width
        uint64_t pr_miss[2]    = {};
        uint64_t pr_cov[2][3]  = {};
        uint64_t pr_up[2][3]   = {};
    };

    int  layer_idx(int il);
    std::vector<int32_t> table_slots() const;
    std::vector<int32_t> table_ins()   const;
    void sim_init(sim & s) const;
    // on_step: called at the start of each step with the slots as the graph of the step reads them
    void replay_layer    (sim & s, int idx, const std::function<void(size_t k, int32_t g, const sim_layer & l)> * on_step = nullptr) const;
    void replay_layer_opt(sim & s, int idx) const;
    void replay_pooled   (sim & s) const;
    void replay_pred();
    void replay();

    const llama_model & model;

    int64_t n_expert      = 0;
    int64_t n_expert_used = 0;

    uint32_t period       = 0;
    uint32_t n_tokens_dec = 0; // a step with at most this many tokens counts as decode

    int32_t cache_slots = 0; // --moe-cache, 0 when there is no cache
    int32_t cache_ins   = 0;
    bool    cache_lfu   = false;

    uint64_t n_steps_dec  = 0;
    uint64_t n_steps_all  = 0;
    uint64_t n_steps_rep  = 0; // decode steps at the last report
    uint64_t n_tok_dec    = 0;

    int64_t  t_replay_us  = 0; // time of the last replay

    std::vector<int>       il2idx;
    std::vector<moe_layer> mlayers;
    std::vector<sim>       sims;

    // the prefetch by prediction (t_moe_pred): top ids per token, the widths it tries and the cache it is measured on
    int64_t n_pred_top = 0;
    int32_t pred_top[3] = { 0, 0, 0 };
    sim     pred_sim;

    uint64_t tr_step0 = 0; // decode step at the start of the trace

    std::vector<int32_t>       buf;
    std::vector<ggml_backend_t> backends;
};

// device cache of the MoE experts that stay in host memory (--moe-cache)
// the graph computes the cached experts from device slots and the host mul_mat_id nodes skip them.
// one context owns one cache, the slots are uploaded on a separate backend instance of the device.
class llama_moe_cache_fork {
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

    llama_moe_cache_fork(const llama_model & model, int32_t n_slots_req);
    ~llama_moe_cache_fork();

    // select the layers and create the upload backend, called once before alloc()
    void init(ggml_backend_sched_t sched);

    // allocate the slots from the free device memory and upload the experts that the last release() kept. the layers
    // that kept their slots stay as they are, the others get as many as fit and at most as many as those have
    void alloc();

    // free the slots, keep the cached experts of each layer in LRU order for the next alloc()
    void release();

    // free the slots of whole layers until at least need bytes are free, the layers with the most slots first (so that
    // the counts even out), and return the bytes freed. the other layers keep their slots and experts, their finished
    // uploads readable at once: a prompt phase takes the memory its buffers need and the next alloc() uploads only the
    // layers it took (25.10). need 0 frees nothing
    size_t release_bytes(size_t need);

    // device bytes of the slots of all layers
    size_t size_slots() const;

    // the experts from e on of the host weight w that the slots hold, while their slots follow each other (up to last),
    // copied into dst in device memory on backend; returns how many, 0 when the slots do not hold expert e. the copy of
    // the split inputs uploads the rest (llama_context::sched_copy_experts, as for the upstream cache)
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    // all of the host weight w into dst on backend: the experts the slots hold from device memory, the others uploaded
    // (the uploads ahead of GGML_SCHED_PREFETCH); false when w is no weight of a layer with slots on this device, then
    // the caller uploads it all. a prompt phase keeps the slots it leaves room for, so their experts need no PCIe
    bool upload_weight(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst);

    bool is_init() const { return initialized; }
    bool ready()   const { return n_live > 0; }

    int32_t get_n_slots() const { return n_slots_l; }
    int32_t get_max_ins() const { return max_ins; }
    bool uses_frequency_admission() const { return frequency_admission; }

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
        std::vector<uint16_t> freq;

        std::vector<int32_t> uploads_new;  // slots with an upload that the last update() started
        std::vector<int32_t> uploads_done; // slots with an upload that finished, but is not published yet
        std::vector<int32_t> evicted;      // published experts that lost their slot

        int32_t head    = -1;
        int32_t tail    = -1;
        int32_t n_fill  = 0;
        int32_t n_seen  = 0;
        int32_t n_slots = 0; // slots of the layer, 0 while it has none
    };

    void   lru_unlink(lru & c, int32_t e);
    void   lru_push  (lru & c, int32_t e);
    void   insert    (size_t idx, int32_t e);
    size_t upload    (size_t idx, int32_t e, int32_t s);
    void   join_uploads();

    // the experts each layer held, kept across runs in LLAMA_MOE_CACHE_STATE (a file path): saved when the slots are
    // released and when the cache goes away, read by init() so that the first alloc() uploads them
    void   state_load();
    void   state_save() const;
    std::vector<std::vector<int32_t>> hot_now() const;

    const llama_model & model;

    const int32_t n_slots_req;

    bool initialized  = false;
    bool alloc_logged = false; // the first alloc() logs at warn level
    bool frequency_admission = false;
    bool async_upload = false;
    bool async_table = false;

    struct pending_upload {
        size_t idx;
        int32_t expert;
        int32_t slot;
    };
    std::vector<pending_upload> pending_uploads;
    std::thread upload_thread;
    int64_t upload_submit_us = 0; // read only after the worker is joined
    int64_t t_submit_us = 0;

    int64_t n_expert  = 0;
    int32_t n_live    = 0; // layers with slots
    int32_t n_slots_l = 0; // slots per layer of the last alloc()
    int32_t max_ins   = 1; // max expert uploads per layer per step, 1 measured best up to 192 slots
    size_t  b_slot    = 0; // bytes of one slot on every cached layer
    int     n_moe     = 0; // MoE layers of the model

    ggml_backend_dev_t dev        = nullptr;
    ggml_backend_t     backend    = nullptr; // compute backend of dev in the scheduler, not owned
    ggml_backend_t     backend_up = nullptr; // separate instance for the uploads

    // starts the queued uploads on backends that wait for a submit, nullptr when the device has no events
    ggml_backend_event_t ev_up = nullptr;

    bool ev_up_recorded = false;

    ggml_context_ptr        ctx_dev; // the table
    ggml_context_ptr        ctx_host;
    ggml_backend_buffer_ptr buf_dev;
    ggml_backend_buffer_ptr buf_host;

    // the slots of each layer in a buffer of their own, so that some layers can give them up and the rest keep them
    std::vector<ggml_context_ptr>        ctx_slots;
    std::vector<ggml_backend_buffer_ptr> buf_slots;

    size_t slot_bytes(int idx) const; // one slot of the layer

    // LLAMA_MOE_CACHE_COPY (25.10, 1; 0: the prompt copies upload every expert from the host, as before)
    bool copy_dev = true;

    std::unordered_map<const ggml_tensor *, std::pair<int, int>> bound; // host weight -> layer, 0 up 1 gate 2 down
    ggml_context_ptr ctx_views; // of copy_run()

    bool    bound_live(ggml_backend_t backend, const ggml_tensor * w, int & idx, int & which) const;
    int64_t copy_run  (ggml_backend_t backend, int idx, int which, ggml_tensor * dst, int64_t e, int64_t last);

    // LLAMA_MOE_CACHE_FILL=<MiB> (25.10, 0): alloc() gives the experts of the last release() their slots, and the steps
    // upload them in the background, at most this much per step, the most recent of every layer first; until then
    // the graphs compute them on the host. 0: alloc() uploads them all before the next graph
    size_t fill_bytes = 0;

    struct fill_entry {
        int     idx;
        int32_t expert;
        int32_t slot;
    };
    std::vector<fill_entry> fill_q;   // waiting, from fill_pos on
    size_t                  fill_pos = 0;

    bool fill_issue(); // the next uploads of fill_q within fill_bytes, true when it started some

    uint64_t n_fill_up = 0; // experts that the background fill uploaded

    uint64_t n_copy_dev = 0; // experts that the prompt copies took from the slots
    uint64_t b_copy_dev = 0;
    uint64_t b_copy_up  = 0; // what the same copies uploaded
    void   wait_idle();               // the graph and the uploads are done with the slots
    void   publish_done();            // the finished uploads become readable
    void   free_all();                // the slots and the table of all layers

    ggml_tensor * table_all = nullptr; // device I32 [n_expert, n_layers]
    ggml_tensor * table_staging = nullptr;


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
