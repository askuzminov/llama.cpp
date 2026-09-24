#pragma once

#include "llama-memory-hybrid.h"

#include <map>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
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

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

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

    // as llama_memory_hybrid, plus the indexer: it must be restored into the very cells the attention
    // cache used, so the delta path carries the slot layout across just like state_read does
    size_t state_write_delta(
            llama_io_write_i & io,
            llama_seq_id seq_id,
            llama_state_seq_flags flags,
            llama_pos base_pos) const override;

    bool state_read_delta(
            llama_io_read_i  & io,
            llama_seq_id seq_id,
            llama_state_seq_flags flags,
            llama_pos base_pos) override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to, null if the caller has no use for it
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block, unpooled cells included, may be null
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns], -inf on any block
    // the query cannot see at all. a block the query sees only in part stays finite, so the caller
    // still has to add the attention mask.
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * bias,
                       const llama_ubatch * ubatch, uint32_t n_kv, uint32_t ratio, bool blk_bias) const;

    // what one pass has to rebuild of the pooled block keys
    struct qsa_plan {
        uint32_t n_dirty = 0;   // rows the graph writes; the list is padded to this size
        bool     warm    = false;
    };

    // group the cells into blocks and compare them with the blocks the pool already holds.
    // this runs at graph build time, so the graph can size its dirty list before set_input_qsa
    // fills it. the result is kept until the next ubatch, so a second call costs nothing.
    qsa_plan qsa_prepare(const llama_ubatch * ubatch, uint32_t n_kv, uint32_t ratio, bool blk_bias,
                         uint32_t s0, uint32_t ns) const;

    // cells and rows of the blocks qsa_prepare marked dirty, and their mrope positions
    //   dirty_cells I32 [ratio*n_dirty, ns]
    //   dirty_pos   I32 [4*n_dirty*ns]
    //   dirty_rows  I32 [n_dirty, ns]
    void set_input_qsa_dirty(ggml_tensor * dirty_cells, ggml_tensor * dirty_pos, ggml_tensor * dirty_rows,
                             const llama_ubatch * ubatch, uint32_t ratio) const;

    // pooled, normed and rotated block keys of one layer, kept across passes:
    // F32 [indexer_head_size, kv_size/ratio, n_stream]. null when the model has no indexer
    ggml_tensor * get_pool(int32_t il) const;

    // the pool holds derived data, so nothing restores it: every cell move drops it
    void qsa_pool_drop();

    // a new ubatch invalidates what qsa_prepare cached
    void qsa_step() const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

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
    };

    struct qsa_state {
        uint64_t epoch    = 0;            // the ubatch this was computed for
        uint32_t n_kv     = 0;
        int64_t  n_blocks = 0;
        uint32_t n_dirty  = 0;
        bool     blk_bias = false;
        bool     warm     = false;
        std::vector<qsa_group> grp;       // per ubatch stream
    };

    // the block tables each cache stream's pool rows were built from
    struct qsa_pool_stream {
        std::vector<int32_t> cells;       // ratio*n
        std::vector<int32_t> pos;         // 4*n
        int32_t n = 0;
    };

    // layers can differ in ratio, so everything below is kept per ratio
    mutable std::map<uint32_t, qsa_state> qsa_cur;
    mutable std::map<uint32_t, std::vector<qsa_pool_stream>> qsa_pool;

    mutable uint64_t qsa_epoch = 1;

    std::map<int32_t, ggml_tensor *> pool_tensors;

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> pool_bufs;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * bias,
                       const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;

    // see llama_memory_hybrid_idx::qsa_prepare. a context with no ubatch of its own asks for the
    // worst case, so that the reserved graph covers every later one
    llama_memory_hybrid_idx::qsa_plan qsa_prepare(const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;

    void set_input_qsa_dirty(ggml_tensor * dirty_cells, ggml_tensor * dirty_pos, ggml_tensor * dirty_rows,
                             const llama_ubatch * ubatch, uint32_t ratio) const;

    // this ubatch's streams of the stored block keys: F32 [indexer_head_size, kv_size/ratio, ns]
    ggml_tensor * get_pool(ggml_context * ctx, int32_t il) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // a full-cache context groups nothing, so it cannot say what is dirty
    bool is_full = false;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
