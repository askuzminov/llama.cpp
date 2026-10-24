#pragma once

#include "llama-memory-hybrid.h"

#include <limits>
#include <map>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx_fork
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

// TODO: this memory module is pending complete reimplementation - do not use for model other than Qwen4

class llama_memory_hybrid_idx_fork : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx_fork(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    // reports the comparisons of LLAMA_QSA_CHECK
    ~llama_memory_hybrid_idx_fork();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void set_n_kv_full(uint32_t n_kv) override;

    // the elastic KV cache: the attention cache searches its capacity, the indexer cache maps memory for the same cells
    bool     kv_elastic_init_fork(bool emulate) override;
    uint32_t kv_capacity_fork() const override;
    uint32_t kv_set_capacity_fork(uint32_t n_cells) override;
    size_t   kv_cell_bytes_fork() const override;
    void     kv_set_grow_fork(std::function<bool(uint32_t)> grow) override;

    // the pool rows of a failed pass may be half written
    void ubatch_failed() override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx_fork specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    bool is_kv_host() const { return kv_host; }

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to, null if the caller has no use for it
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block, unpooled cells included, may be null
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns], -inf on any block
    // the query cannot see at all. a block the query sees only in part stays finite, so the caller
    // still has to add the attention mask.
    // causal_attn selects the rule: causal forces the query's own block on, non-causal lets every visible block compete on score
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * bias,
                       const llama_ubatch * ubatch, uint32_t n_kv, uint32_t ratio, bool blk_bias,
                       bool causal_attn) const;

    // the attention mask of QSA at the cells top_k picks, computed on the device from positions instead of read out
    // of the dense KQ mask (LLAMA_QSA_MASK_POS): for a ubatch of tokens, causal, without ALiBi or SWA, cell j is visible
    // to a token when the token's sequence holds it and its temporal position is not above the token's, as
    // set_input_kq_mask decides. with one sequence per stream:
    //   cell_pos F32 [1, n_kv, 1, ns]          position of each attention cell of the stream's sequence, +inf otherwise
    //   tok_pos  F32 [1, 1, n_tokens/ns, ns]   position of each token
    // with several sequences in one stream (tok_half given: the slots of a unified cache in one ubatch, 23.10), each
    // sequence of the stream gets its own range of positions, rank*2^QSA_POS_STRIDE_LOG2 on, so a cell is visible when it
    // lies in [rank*stride, rank*stride + pos] of the token, i.e. |cell - mid| <= half:
    //   cell_pos F32                           rank of the cell's sequence * stride + its position, +inf when empty
    //   tok_pos  F32                           mid  = rank of the token's sequence * stride + pos/2
    //   tok_half F32 [1, 1, n_tokens/ns, ns]   half = pos/2
    // a cell that two sequences hold has no single range: qsa_pos_multi_ok then says no and the dense mask is used
    void set_input_qsa_pos(ggml_tensor * cell_pos, ggml_tensor * tok_pos, ggml_tensor * tok_half, const llama_ubatch * ubatch,
                           uint32_t n_kv) const;

    // LLAMA_QSA_POS_STRIDE_LOG2 lowers it for a test (qsa_pos_stride_log2)
    static constexpr int QSA_POS_STRIDE_LOG2 = 18;

    // whether the ranges of set_input_qsa_pos can describe this ubatch's stream: no cell held by two sequences, every
    // position below the stride and all the values exact in f32 (up to 32 sequences). checked once per ubatch
    bool qsa_pos_multi_ok(const llama_ubatch * ubatch, uint32_t n_kv) const;

    // what one pass has to rebuild of the pooled block keys
    struct qsa_plan {
        uint32_t n_dirty = 0;   // rows the graph writes; the list is padded to this size
        bool     warm    = false;
    };

    // group the cells into blocks and compare them with the blocks the pool already holds.
    // this runs at graph build time, so the graph can size its dirty list before set_input_qsa
    // fills it. the result is kept until the next ubatch, so a second call costs nothing.
    // sinfo names the cells of this ubatch, so that the grouping of the last ubatch can be carried forward
    qsa_plan qsa_prepare(const llama_ubatch * ubatch, uint32_t n_kv, uint32_t ratio, bool blk_bias,
                         uint32_t s0, uint32_t ns, const llama_kv_cache::slot_info * sinfo = nullptr) const;

    // cells and rows of the blocks qsa_prepare marked dirty, and their mrope positions
    //   dirty_cells I32 [ratio*n_dirty, ns]
    //   dirty_pos   I32 [4*n_dirty*ns]
    //   dirty_rows  I32 [n_dirty, ns]
    void set_input_qsa_dirty(ggml_tensor * dirty_cells, ggml_tensor * dirty_pos, ggml_tensor * dirty_rows,
                             uint32_t ratio) const;

    // pooled, normed and rotated block keys of one layer, kept across passes:
    // F32 [indexer_head_size, kv_size/ratio, n_stream]. null when the model has no indexer
    ggml_tensor * get_pool(int32_t il) const;

    // the block scores of the first draft step of an MTP block, before the bias: F32 [kv_size/ratio, n_stream].
    // null for a trunk layer
    ggml_tensor * get_score(int32_t il) const;

    // the pool holds derived data, so nothing restores it: every cell move drops it
    void qsa_pool_drop() const;

    // drop only the pooled blocks that hold a cell the removal takes. a rollback cuts the tail,
    // so the blocks before it keep their cells and their keys
    void qsa_pool_trim(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;

    // a new ubatch invalidates what qsa_prepare cached
    void qsa_step() const;

private:
    bool kv_host = false;
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // the indexer cache is elastic too (kv_elastic_init_fork)
    bool idx_elastic = false;

    //
    // qwen4exp QSA: block keys pooled from the indexer cells, kept across passes
    //

    // the grouping of one ubatch stream, as qsa_prepare left it
    struct qsa_group {
        std::vector<int32_t> blk_cells;   // ratio*n_blocks
        std::vector<int32_t> blk_pos;     // 4*n_blocks
        std::vector<int32_t> blk_of;      // n_kv, empty when the bias is per block
        std::vector<int32_t> bid_idx;
        std::vector<int32_t> bid_cell;
        std::vector<llama_kv_cells::seq_set_t> spare_seq;
        std::vector<int32_t> order;       // cells by mrope rank, empty unless ranked
        std::vector<int32_t> rank;
        int32_t  n_bid  = 0;
        int32_t  n_end  = 0;              // blocks in use: pooled ones plus the spare ones after them
        int32_t  n_keep = 0;              // leading blocks the pool already holds
        uint32_t strm   = 0;              // cache stream this ubatch stream sits in
        bool     ranked = false;

        // carried forward to the next ubatch by qsa_group_update: one sequence seq on positions 0.., full blocks in
        // order, n_spare cells of the incomplete tail. dirty_pos: a rollback took the positions from there on
        bool         inc_ok    = false;
        llama_seq_id seq       = -1;
        int32_t      n_spare   = 0;
        int32_t      n_same    = 0;       // leading blocks the last update left as they were
        llama_pos    dirty_pos = std::numeric_limits<llama_pos>::max();

        // carried forward by qsa_group_update_multi: several sequences, one per cell, the groups of the last
        // grouping keyed on (sequence, bucket) and the tail rollbacks since: m_cut[q] the first position a
        // rollback of q took, m_cut_hi[q] the last position q had then
        bool                   inc_multi = false;
        uint64_t               m_epoch   = 0;       // the ubatch the groups describe
        int64_t                m_n_cells = 0;       // the cells in groups
        bool                   m_any_cut = false;
        std::vector<int32_t>   m_cell_grp;          // n_kv: the group of a cell, -1 for none
        std::vector<int32_t>   m_head;              // per bucket: the first group of its chain
        std::vector<int32_t>   m_next;              // per group
        std::vector<int32_t>   m_seq;               // per group: its sequence
        std::vector<int32_t>   m_pb;                // per group: its bucket
        std::vector<uint64_t>  m_slots;             // per group: the slots it has a cell for
        std::vector<int32_t>   m_cell;              // ratio per group: the cell of each slot, -1 for none
        std::vector<int32_t>   m_bid;               // per group: its block id, -1 for none
        std::vector<uint8_t>   m_in_open;           // per group: listed in m_open
        std::vector<int32_t>   m_bid_grp;           // per block id: its group
        std::vector<int32_t>   m_open;              // the groups that may be incomplete
        std::vector<llama_pos> m_cut;               // per sequence
        std::vector<llama_pos> m_cut_hi;
    };

    // regroup only the blocks of g that can have changed since the last ubatch: those from the first incomplete
    // one, the first one a rollback took, or the first one a new cell falls into. false when g cannot be carried,
    // the caller then groups every cell
    bool qsa_group_update(qsa_group & g, const llama_kv_cells & cells, const std::vector<uint32_t> & new_cells,
                          int64_t n_kv, int64_t r) const;

    // the same for several sequences in one stream: the rollbacks since the last ubatch, then its new cells. the
    // blocks keep their ids up to the first one a rollback took, the ones after it keep their order, and the blocks
    // completed now follow by their last cell, as the full grouping orders them (group_full). false when g cannot
    // be carried. on the Mac, 4 slots at 8K each: 13 us per decode pass against 1260 for the full grouping
    bool qsa_group_update_multi(qsa_group & g, const llama_kv_cells & cells, const std::vector<uint32_t> & new_cells,
                                int64_t n_kv, int64_t r) const;

    mutable std::vector<int32_t> qsa_scratch;

    struct qsa_state {
        uint64_t epoch    = 0;            // the ubatch this was computed for
        uint32_t n_kv     = 0;
        int64_t  n_blocks = 0;
        uint32_t n_dirty  = 0;
        bool     blk_bias = false;
        bool     warm     = false;
        std::vector<qsa_group> grp;       // per ubatch stream
    };

    // the block tables each cache stream's pool rows were built from. a trim keeps the rows past n: several
    // sequences order their blocks by them (group_full)
    struct qsa_pool_stream {
        std::vector<int32_t> cells;       // ratio per row, at least n rows
        std::vector<int32_t> pos;         // 4 per row
        int32_t n = 0;                    // the rows that hold the keys of their blocks
    };

    // layers can differ in ratio, so everything below is kept per ratio
    mutable std::map<uint32_t, qsa_state> qsa_cur;
    mutable std::map<uint32_t, std::vector<qsa_pool_stream>> qsa_pool;

    mutable uint64_t qsa_epoch = 1;

    // qsa_pos_multi_ok of the ubatch of qsa_pos_ok_epoch
    mutable uint64_t qsa_pos_ok_epoch = 0;
    mutable uint32_t qsa_pos_ok_n_kv  = 0;
    mutable bool     qsa_pos_ok       = false;

    std::map<int32_t, ggml_tensor *> pool_tensors;
    std::map<int32_t, ggml_tensor *> score_tensors;

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> pool_bufs;
};

class llama_memory_hybrid_idx_fork_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_fork_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_fork_context(llama_memory_hybrid_idx_fork * mem);

    // used to create an update context
    llama_memory_hybrid_idx_fork_context(
            llama_memory_hybrid_idx_fork * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_fork_context(
            llama_memory_hybrid_idx_fork * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_fork_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_fork_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // the ubatch after the current one, nullptr for the last (the PLE rows of qwen4exp are gathered while the current
    // one computes, see llama_fork_before_compute)
    const llama_ubatch * peek_next() const { return i_cur + 1 < ubatches_peek.size() ? &ubatches_peek[i_cur + 1] : nullptr; }

    // the current ubatch is the first of its decode call
    bool is_first() const { return i_cur == 0; }

    bool is_kv_host() const { return mem && mem->is_kv_host(); }

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * bias,
                       const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias, bool causal_attn) const;

    // see llama_memory_hybrid_idx_fork::qsa_prepare. a context with no ubatch of its own asks for the
    // worst case, so that the reserved graph covers every later one
    llama_memory_hybrid_idx_fork::qsa_plan qsa_prepare(const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;

    void set_input_qsa_dirty(ggml_tensor * dirty_cells, ggml_tensor * dirty_pos, ggml_tensor * dirty_rows,
                             uint32_t ratio) const;

    // see llama_memory_hybrid_idx_fork::set_input_qsa_pos, over the attention cells of this ubatch
    void set_input_qsa_pos(ggml_tensor * cell_pos, ggml_tensor * tok_pos, ggml_tensor * tok_half, const llama_ubatch * ubatch) const;

    // see llama_memory_hybrid_idx_fork::qsa_pos_multi_ok
    bool qsa_pos_multi_ok(const llama_ubatch * ubatch) const;

    // this ubatch's streams of the stored block keys: F32 [indexer_head_size, kv_size/ratio, ns]
    ggml_tensor * get_pool(ggml_context * ctx, int32_t il) const;

    // this ubatch's streams of the saved block scores of an MTP block: F32 [kv_size/ratio, ns], null for a trunk layer
    ggml_tensor * get_score(ggml_context * ctx, int32_t il) const;

private:
    const llama_memory_hybrid_idx_fork * mem = nullptr;

    // a full-cache context groups nothing, so it cannot say what is dirty
    bool is_full = false;

    // an update context shifts or copies cells, so applying it drops the pool
    bool is_update = false;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // the ubatches again, for peek_next(): the base class keeps its copy private
    const std::vector<llama_ubatch> ubatches_peek;
};
