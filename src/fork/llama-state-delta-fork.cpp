// the fork's delta state of a sequence: the cells added after a base position, written and read on top of a full
// state that was loaded before (context checkpoints of the server). llama-memory.h and the cache headers declare
// these members

#include "llama-memory.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

size_t llama_memory_i::state_write_delta(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        llama_pos base_pos) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    GGML_UNUSED(base_pos);
    return 0;
}

bool llama_memory_i::state_read_delta(
        llama_io_read_i  & io_delta,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        llama_pos base_pos) {
    GGML_UNUSED(io_delta);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    GGML_UNUSED(base_pos);
    return false;
}

//
// Delta state write/read for KV cache
//

// a delta checkpoint starts with this, so a full state can never be mistaken for one
static constexpr uint32_t KV_DELTA_MAGIC = 0x4B56444C; // "KVdL"

size_t llama_kv_cache::state_write_delta(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        llama_pos base_pos) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    // a mirrored cache writes nothing in state_write, so it has no delta of its own either
    if (other) {
        return 0;
    }

    // the reader restores into a single sequence, so a whole-context delta has no way back
    if (seq_id < 0) {
        return 0;
    }

    GGML_UNUSED(flags);

    GGML_ASSERT((size_t) seq_id < seq_to_stream.size());

    const size_t start_bytes = io.n_bytes();

    const uint32_t magic = KV_DELTA_MAGIC;
    io.write(&magic, sizeof(magic));

    io.write(&base_pos, sizeof(base_pos));
    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count = 0;

        // only the cells added after base_pos: everything up to it is in the base checkpoint
        const cell_ranges_t cr = state_ranges(s, seq_id, base_pos, cell_count);

        io.write(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }

    return io.n_bytes() - start_bytes;
}

bool llama_kv_cache::state_read_delta(
        llama_io_read_i  & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        llama_pos base_pos) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    // nothing was written for a mirrored cache, so there is nothing to restore
    if (other) {
        return true;
    }

    GGML_UNUSED(flags);

    uint32_t delta_magic;
    io.read(&delta_magic, sizeof(delta_magic));

    if (delta_magic != KV_DELTA_MAGIC) {
        LLAMA_LOG_ERROR("%s: invalid delta magic: 0x%08X\n", __func__, delta_magic);
        return false;
    }

    llama_pos delta_base_pos;
    io.read(&delta_base_pos, sizeof(delta_base_pos));

    if (delta_base_pos != base_pos) {
        LLAMA_LOG_ERROR("%s: base_pos mismatch: checkpoint has %d, expected %d\n",
                        __func__, (int) delta_base_pos, (int) base_pos);
        return false;
    }

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));

    if (n_stream_cur != n_stream) {
        LLAMA_LOG_ERROR("%s: stream count mismatch\n", __func__);
        return false;
    }

    // must match the write side, which refuses to produce a whole-context delta
    if (seq_id < 0) {
        LLAMA_LOG_ERROR("%s: whole-cache delta restore is not supported\n", __func__);
        return false;
    }

    GGML_ASSERT((size_t) seq_id < seq_to_stream.size());

    const uint32_t strm = seq_to_stream[seq_id];

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        slot_info sinfo;

        // clear_seq = false: the delta adds to the cells the base checkpoint already put in place
        bool res = state_read_meta(io, strm, cell_count, sinfo, seq_id, nullptr, false);

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);
        } catch (...) {
            res = false;
        }

        if (!res) {
            // half a delta is neither the base nor the checkpoint - drop the sequence, the caller recomputes it
            state_clear(seq_id, strm, sinfo);

            return false;
        }
    }

    return true;
}

size_t llama_kv_cache_iswa::state_write_delta(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        llama_pos base_pos) const {
    const size_t n_bytes_start = io.n_bytes();

    // must mirror state_write(): the base cache is skipped for a partial checkpoint
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (kv_base->state_write_delta(io, seq_id, flags, base_pos) == 0) {
            return 0; // sub-cache cannot produce a delta
        }
    }

    if (kv_swa->state_write_delta(io, seq_id, flags, base_pos) == 0) {
        return 0;
    }

    return io.n_bytes() - n_bytes_start;
}

bool llama_kv_cache_iswa::state_read_delta(
        llama_io_read_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        llama_pos base_pos) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (!kv_base->state_read_delta(io, seq_id, flags, base_pos)) {
            return false;
        }
    }

    return kv_swa->state_read_delta(io, seq_id, flags, base_pos);
}
