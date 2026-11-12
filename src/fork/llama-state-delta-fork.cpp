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

// the magic and the base position open the delta of every cache: of a plain one, of the first of an iSWA pair
llama_pos llama_state_seq_delta_base_pos(const uint8_t * src, size_t size) {
    uint32_t  magic    = 0;
    llama_pos base_pos = -1;
    if (src == nullptr || size < sizeof(magic) + sizeof(base_pos)) {
        return -1;
    }
    memcpy(&magic, src, sizeof(magic));
    if (magic != KV_DELTA_MAGIC) {
        return -1;
    }
    memcpy(&base_pos, src + sizeof(magic), sizeof(base_pos));
    return base_pos >= 0 ? base_pos : -1;
}

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

    // must mirror state_write(): a partial state of a plain cache is empty, so there is no partial delta either
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) && swa_type == LLAMA_SWA_TYPE_NONE) {
        return 0;
    }

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

    // must mirror state_read(): nothing to read for a partial state of a plain cache
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) && swa_type == LLAMA_SWA_TYPE_NONE) {
        return true;
    }

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

    // the delta of one sequence holds the cells of the one stream it lived in: it goes to the stream of the sequence here
    // whatever the stream count of the cache it came from (state_read_sinfo does the same for the whole state)
    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    uint32_t n_filled = 0;

    // must match the write side, which refuses to produce a whole-context delta
    if (seq_id < 0) {
        LLAMA_LOG_ERROR("%s: whole-cache delta restore is not supported\n", __func__);
        return false;
    }

    GGML_ASSERT((size_t) seq_id < seq_to_stream.size());

    const uint32_t strm = seq_to_stream[seq_id];

    for (uint32_t s = 0; s < n_stream_cur; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }
        // also with the same stream count (review 10.11): the cells of several streams would all go into strm
        if (++n_filled > 1) {
            LLAMA_LOG_ERROR("%s: the delta of one sequence holds cells of several streams\n", __func__);
            return false;
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
