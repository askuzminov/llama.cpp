#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    if (!mem_idx) {
        return;
    }

    // block keys pooled from the indexer cells. they only change where the cells change, so they
    // are kept across passes instead of being rebuilt from the whole cache on every one
    const llama_hparams & hp = model.hparams;

    const int64_t  idx_dim  = hp.indexer_head_size;
    const uint32_t n_cells  = mem_idx->get_size();
    const uint32_t n_stream = mem_idx->get_n_stream();

    if (idx_dim == 0) {
        return;
    }

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    for (uint32_t il = 0; il < hp.n_layer_all; il++) {
        if (filter_idx && !filter_idx(il)) {
            continue;
        }

        const uint32_t r = hp.dsv4_compress_ratios[il];

        if (r == 0) {
            continue;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
        }

        auto it = ctx_map.find(buft);

        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(hp.n_layer_all*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context for the qsa pool");
            }

            it = ctx_map.emplace(buft, ctx).first;
        }

        ggml_tensor * t = ggml_new_tensor_3d(it->second.get(), GGML_TYPE_F32, idx_dim, (n_cells + r - 1)/r, n_stream);

        ggml_format_name(t, "qsa_pool_l%d", il);

        pool_tensors[il] = t;
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;

        if (hp.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }

        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for the qsa pool");
        }

        LLAMA_LOG_INFO("%s: %10s QSA pool size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        // a row past the last pooled block is still read, and a nan there would beat a real block
        // in the top-k. it is never read for its value, so any finite number does
        ggml_backend_buffer_clear(buf, 0);

        pool_bufs.emplace_back(std::move(ctx), buf);
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    // note: the pool is dropped when the context is applied, not here: llama_context::decode asks
    //       for an update on every call and most of them turn out to be no-ops
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    qsa_pool_drop();

    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    qsa_pool_drop();

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    qsa_pool_drop();

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    qsa_pool_drop();

    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    qsa_pool_drop();

    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    qsa_pool_drop();

    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    for (const auto & [_, buf] : pool_bufs) {
        mb[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    qsa_pool_drop();

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

size_t llama_memory_hybrid_idx::state_write_delta(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        llama_pos base_pos) const {
    const size_t n_bytes_start = io.n_bytes();

    if (llama_memory_hybrid::state_write_delta(io, seq_id, flags, base_pos) == 0) {
        return 0; // the attention cache cannot produce a delta
    }

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx && mem_idx->state_write_delta(io, seq_id, flags, base_pos) == 0) {
            return 0;
        }
    }

    return io.n_bytes() - n_bytes_start;
}

bool llama_memory_hybrid_idx::state_read_delta(
        llama_io_read_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        llama_pos base_pos) {
    // note: repeats llama_memory_hybrid::state_read_delta
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO] the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    bool res = true;

    qsa_pool_drop();

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            res = get_mem_attn()->state_read_delta_sinfo(io, seq_id, flags, base_pos, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        if (res) {
            // recurrent full state (mem_recr->state_read throws on failure)
            get_mem_recr()->state_read(io, seq_id, flags);
        }

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write_delta
        if (res && mem_idx && (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            res = mem_idx->state_read_delta_sinfo(io, seq_id, flags, base_pos, nullptr, &sinfos_attn);
        }
    } catch (...) {
        state_drop(seq_id);

        throw;
    }

    if (!res) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        state_drop(seq_id);
    }

    return res;
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    qsa_pool_drop();

    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

//
// qwen4exp QSA: pooled block keys
//

ggml_tensor * llama_memory_hybrid_idx::get_pool(int32_t il) const {
    const auto it = pool_tensors.find(il);

    return it == pool_tensors.end() ? nullptr : it->second;
}

void llama_memory_hybrid_idx::qsa_pool_drop() const {
    // only the bookkeeping is dropped. the rows keep their numbers, which is all the blocks past
    // the last pooled one need: their bias decides them, not their score
    qsa_pool.clear();
    qsa_cur .clear();
}

void llama_memory_hybrid_idx::qsa_step() const {
    qsa_epoch++;
}

llama_memory_hybrid_idx::qsa_plan llama_memory_hybrid_idx::qsa_prepare(
        const llama_ubatch * ubatch,
        uint32_t n_kv_arg,
        uint32_t ratio,
        bool blk_bias,
        uint32_t s0,
        uint32_t ns) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    qsa_state & st = qsa_cur[ratio];

    // the graph asks once to size its inputs and once more to check it can be reused
    if (st.epoch == qsa_epoch && st.n_kv == n_kv_arg && st.blk_bias == blk_bias) {
        return { st.n_dirty, st.warm };
    }

    const int64_t n_kv     = n_kv_arg;
    const int64_t n_ns     = ns;
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;
    const int64_t n_blocks = (n_kv + r - 1)/r;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    st.epoch    = qsa_epoch;
    st.n_kv     = n_kv_arg;
    st.n_blocks = n_blocks;
    st.blk_bias = blk_bias;
    st.grp.resize(n_ns);

    auto & pool = qsa_pool[ratio];
    pool.resize(get_mem_idx()->get_n_stream());

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_slot0;

    int n_seq_max_seen = 1;

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        qsa_group & g = st.grp[s];

        g.strm = s0 + s;

        g.blk_cells.assign(r*n_blocks, 0);
        g.blk_pos  .assign(4*n_blocks, 0);
        g.blk_of   .assign(blk_bias ? 0 : n_kv, -1);

        g.bid_idx  .clear();
        g.bid_cell .clear();
        bid_slot0  .clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        n_seq_max_seen = std::max(n_seq_max_seen, n_seq_present);

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        g.ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = g.ranked ? g.rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t gi = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        gi = c;
                        break;
                    }
                }

                if (gi < 0) {
                    gi = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = gi;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[gi] & bit) != 0;

                cell_grp[j]    = gi;
                grp_slots[gi] |= bit;

                if (idx%r == 0) {
                    grp_slot0[gi] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            g.order.clear();
            g.order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    g.order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(g.order.begin(), g.order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            g.rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) g.order.size(); ++k) {
                g.rank[g.order[k]] = (int32_t) k;
            }

            g.ranked = true;

            group_cells();
        } else {
            g.order.clear();
            g.rank .clear();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t gi = grp_head[pb]; gi >= 0; gi = grp_next[gi]) {
                if (grp_slots[gi] != slots_full) {
                    continue;
                }

                grp_bid[gi] = n_bid++;

                g.bid_idx  .push_back((int32_t) (pb*r));
                g.bid_cell .push_back(grp_first[gi]);
                bid_slot0  .push_back(grp_slot0[gi]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        g.n_bid = n_bid;

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { g.bid_idx[b], g.bid_idx[b], g.bid_idx[b], g.bid_idx[b] };

            if (g.ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                g.blk_pos[sec*n_blocks + b] = sec_pos[sec];
            }
        }

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t gi  = cell_grp[j];
            const int32_t bid = gi < 0 ? -1 : grp_bid[gi];

            if (!blk_bias) {
                g.blk_of[j] = bid;
            }

            if (bid >= 0) {
                const int64_t idx = g.ranked ? g.rank[j] : cells.pos_get(j);

                g.blk_cells[bid*r + (idx%r)] = (int32_t) j;
            }
        }

        // blk_cells also has to name the cells no block pools, so that a set of block ids still
        // covers every live cell. they are packed r at a time into the blocks past the pooled
        // ones and always fit: the pooled ones hold exactly n_bid*r cells and n_kv <= n_blocks*r.
        // two cells claiming one slot (duplicate positions across sequences) leave the loser here.
        int64_t n_spare = 0;

        g.spare_seq.assign((size_t) (n_blocks - n_bid), llama_kv_cells::seq_set_t());

        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }

            const int32_t gi  = cell_grp[j];
            const int32_t bid = gi < 0 ? -1 : grp_bid[gi];

            if (bid >= 0) {
                const int64_t idx = g.ranked ? g.rank[j] : cells.pos_get(j);

                if (g.blk_cells[bid*r + (idx%r)] == (int32_t) j) {
                    continue;
                }
            }

            const int64_t sb = n_spare/r;

            GGML_ASSERT(n_bid + sb < n_blocks);

            g.blk_cells[(n_bid + sb)*r + n_spare%r] = (int32_t) j;
            g.spare_seq[sb] |= cells.seq_get_all((uint32_t) j);

            n_spare++;
        }

        // pad the last spare block with one of its own cells: a slot left at zero would hand
        // the block cell 0, which has nothing to do with it
        if (n_spare%r != 0) {
            int32_t * blk = g.blk_cells.data() + (n_bid + n_spare/r)*r;

            std::fill(blk + n_spare%r, blk + r, blk[n_spare%r - 1]);
        }

        // the spare blocks are rebuilt with the pooled ones: they sit right after them, so the
        // two make up one range, and then every block a query can pick holds a current key
        g.n_end = n_bid + (int32_t) ((n_spare + r - 1)/r);

        // how much of the pool still stands. a block is the same block while it pools the same
        // cells at the same place on the position line; the cells themselves cannot change under
        // it, because every op that moves or reuses a cell drops the pool
        const qsa_pool_stream & ps = pool[g.strm];

        int32_t n_keep = std::min(ps.n, n_bid);

        for (int32_t b = 0; b < n_keep; ++b) {
            bool same = true;

            for (int64_t i = 0; i < r && same; ++i) {
                same = ps.cells[b*r + i] == g.blk_cells[b*r + i];
            }

            for (int64_t sec = 0; sec < 4 && same; ++sec) {
                same = ps.pos[b*4 + sec] == g.blk_pos[sec*n_blocks + b];
            }

            if (!same) {
                n_keep = b;
                break;
            }
        }

        g.n_keep = n_keep;
    }

    // a pass can only complete the blocks its own tokens fall into, plus the spare blocks after
    // them: one partial bucket per sequence. this covers the steady state; anything else - a
    // fresh cache, a restored one, a rolled back one - rebuilds the lot.
    // the size must hold still between passes, or the graph is rebuilt every time, so it counts
    // the sequences present rather than the spare cells this pass happens to have
    const uint32_t spare_cap = (uint32_t) ((n_seq_max_seen*(r - 1) + r - 1)/r) + 1;
    const uint32_t cap       = (uint32_t) ((n_tps + r - 1)/r + 1) + spare_cap;

    uint32_t need = 0;

    for (const auto & g : st.grp) {
        need = std::max(need, (uint32_t) (g.n_end - g.n_keep));
    }

    st.warm    = need <= cap;
    st.n_dirty = st.warm ? cap : (uint32_t) n_blocks;

    return { st.n_dirty, st.warm };
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t n_kv_arg,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(bias->buffer && ggml_backend_buffer_is_host(bias->buffer));

    const auto it = qsa_cur.find(ratio);
    GGML_ASSERT(it != qsa_cur.end() && it->second.epoch == qsa_epoch && "qsa: set_input without prepare");

    const qsa_state & st = it->second;

    const int64_t n_kv     = n_kv_arg;
    const int64_t n_ns     = bias->ne[2];            // streams in this ubatch
    const int64_t n_blocks = st.n_blocks;
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(!cell_blk || (cell_blk->ne[0] == n_kv && cell_blk->ne[1] == n_ns));
    GGML_ASSERT(!blk_cells || (blk_cells->ne[0] == r*n_blocks && blk_cells->ne[1] == n_ns));
    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    GGML_ASSERT(st.n_kv == (uint32_t) n_kv && (int64_t) st.grp.size() == n_ns);

    int32_t * dst_cell_blk  = cell_blk ? (int32_t *) cell_blk->data : nullptr;
    int32_t * dst_blk_cells = blk_cells ? (int32_t *) blk_cells->data : nullptr;
    float   * dst_bias      = (float   *) bias->data;

    for (int64_t s = 0; s < n_ns; ++s) {
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        const qsa_group & g = st.grp[s];

        if (dst_blk_cells) {
            std::copy(g.blk_cells.begin(), g.blk_cells.end(), dst_blk_cells + s*(r*n_blocks));
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = g.n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? g.n_bid : (int32_t) n_blocks - 1;

        if (dst_cell_blk) {
            int32_t * cur_cell_blk = dst_cell_blk + s*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                cur_cell_blk[j] = g.blk_of[j] < 0 ? dead_bid : g.blk_of[j];
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (g.ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) g.order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = g.order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it.
                // selection is per block, so whole future blocks drop out here; the block that
                // straddles the query stays and the caller's mask trims its future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < g.n_bid; ++b) {
                    if (g.bid_idx[b] > q || !cells.seq_has((uint32_t) g.bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = g.bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare blocks hold the unpooled cells, the incomplete tail among them, so
                // they get the tail value. one stays finite whenever the sequence has any cell:
                // a sequence shorter than `ratio` owns no full block, and a row of -inf only
                // gives a nan.
                for (int64_t b = g.n_bid; b < n_blocks; ++b) {
                    cur_blk_bias[b] = g.spare_seq[b - g.n_bid].test((size_t) seq_id) ? 1e9f : -INFINITY;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = g.ranked ? g.rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (g.blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

void llama_memory_hybrid_idx::set_input_qsa_dirty(
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_rows,
        const llama_ubatch * ubatch,
        uint32_t ratio) const {
    GGML_UNUSED(ubatch);

    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(dirty_cells->buffer && ggml_backend_buffer_is_host(dirty_cells->buffer));

    const auto it = qsa_cur.find(ratio);
    GGML_ASSERT(it != qsa_cur.end() && it->second.epoch == qsa_epoch && "qsa: set_input without prepare");

    const qsa_state & st = it->second;

    const int64_t r        = ratio;
    const int64_t n_ns     = dirty_rows->ne[1];
    const int64_t n_dirty  = dirty_rows->ne[0];
    const int64_t n_blocks = st.n_blocks;

    GGML_ASSERT((int64_t) st.grp.size() == n_ns);
    GGML_ASSERT(dirty_cells->ne[0] == r*n_dirty && dirty_cells->ne[1] == n_ns);
    GGML_ASSERT(dirty_pos->ne[0] == 4*n_dirty*n_ns);

    // the row numbers address the whole pool view, one stream after the other
    const int64_t n_pool = (get_mem_idx()->get_size() + r - 1)/r;

    int32_t * dst_cells = (int32_t *) dirty_cells->data;
    int32_t * dst_pos   = (int32_t *) dirty_pos  ->data;
    int32_t * dst_rows  = (int32_t *) dirty_rows ->data;

    auto & pool = qsa_pool[ratio];

    for (int64_t s = 0; s < n_ns; ++s) {
        const qsa_group & g = st.grp[s];

        GGML_ASSERT(g.n_end - g.n_keep <= n_dirty);

        // the list is a fixed size, so the tail repeats the last block: writing a row twice with
        // the value it already has costs a row and changes nothing
        for (int64_t i = 0; i < n_dirty; ++i) {
            const int64_t b = std::min<int64_t>(g.n_keep + i, std::max(g.n_end - 1, 0));

            for (int64_t k = 0; k < r; ++k) {
                dst_cells[s*(r*n_dirty) + i*r + k] = g.blk_cells[b*r + k];
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_pos[sec*(n_dirty*n_ns) + s*n_dirty + i] = g.blk_pos[sec*n_blocks + b];
            }

            dst_rows[s*n_dirty + i] = (int32_t) (s*n_pool + b);
        }

        // the graph writes these rows in this pass, so the pool is up to date once it runs
        qsa_pool_stream & ps = pool[g.strm];

        ps.n = g.n_bid;
        ps.cells.assign(g.blk_cells.begin(), g.blk_cells.begin() + (size_t) r*g.n_bid);
        ps.pos.resize((size_t) 4*g.n_bid);

        for (int32_t b = 0; b < g.n_bid; ++b) {
            for (int64_t sec = 0; sec < 4; ++sec) {
                ps.pos[b*4 + sec] = g.blk_pos[sec*n_blocks + b];
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {
    is_full = true;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {
    is_update = true;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    if (mem) {
        // a shift or a stream copy moves cells under the blocks
        if (is_update) {
            mem->qsa_pool_drop();
        }

        // the cells moved, so the grouping qsa_prepare cached no longer describes them
        mem->qsa_step();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);
    GGML_ASSERT(get_idx() != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, bias, ubatch, get_idx()->get_n_kv(), ratio, blk_bias);
}

llama_memory_hybrid_idx::qsa_plan llama_memory_hybrid_idx_context::qsa_prepare(
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);
    GGML_ASSERT(get_idx() != nullptr);
    GGML_ASSERT(ratio > 0);

    const uint32_t n_kv = get_idx()->get_n_kv();

    if (is_full) {
        return { (n_kv + ratio - 1)/ratio, false };
    }

    return mem->qsa_prepare(ubatch, n_kv, ratio, blk_bias, get_idx()->get_s0(), get_n_stream());
}

void llama_memory_hybrid_idx_context::set_input_qsa_dirty(
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_rows,
        const llama_ubatch * ubatch,
        uint32_t ratio) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa_dirty(dirty_cells, dirty_pos, dirty_rows, ubatch, ratio);
}

ggml_tensor * llama_memory_hybrid_idx_context::get_pool(ggml_context * ctx, int32_t il) const {
    GGML_ASSERT(mem != nullptr);
    GGML_ASSERT(get_idx() != nullptr);

    ggml_tensor * pool = mem->get_pool(il);

    if (pool == nullptr) {
        return nullptr;
    }

    const int64_t s0 = get_idx()->get_s0();

    return ggml_view_3d(ctx, pool, pool->ne[0], pool->ne[1], get_n_stream(),
            pool->nb[1], pool->nb[2], s0*pool->nb[2]);
}
