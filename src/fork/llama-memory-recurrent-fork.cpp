// a ubatch that fails on a recurrent state (llama_memory_recurrent::ubatch_save, ubatch_restore), and the memories
// that hold one next to an attention cache

#include "llama-memory-recurrent.h"

#include "llama-batch.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"

#include <algorithm>
#include <vector>

void llama_memory_recurrent::ubatch_save(const llama_ubatch & ubatch) {
    undo.valid   = true;
    undo.cells   = cells;
    undo.head    = head;
    undo.used    = used;
    undo.rs_idx  = rs_idx;
    undo.rec_n   = rec_n;
    undo.rec_set = rec_set;

    // the first token of a sequence holds its lowest position: a batch whose positions go down is refused
    // (llama_batch_allocr::init)
    undo.seqs.clear();
    for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
        const uint32_t i = s*ubatch.n_seq_tokens;
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            undo.seqs.emplace_back(ubatch.seq_id[i][j], ubatch.pos[i]);
        }
    }
}

std::vector<llama_seq_id> llama_memory_recurrent::ubatch_restore(bool started) {
    std::vector<llama_seq_id> lost;
    if (!undo.valid) {
        return lost;
    }
    undo.valid = false;

    // the rows the graph writes are those of the cells of its range: the states of the ubatch, the ones it copies
    // along, and the replay of their records
    const uint32_t row0 = head;
    const uint32_t row1 = started ? head + n : head;

    cells.swap(undo.cells);
    rs_idx.swap(undo.rs_idx);
    rec_n.swap(undo.rec_n);
    rec_set.swap(undo.rec_set);
    head = undo.head;
    used = undo.used;

    const auto lose = [&](llama_seq_id seq_id) {
        if (std::find(lost.begin(), lost.end(), seq_id) == lost.end()) {
            lost.push_back(seq_id);
        }
    };

    for (const auto & [seq_id, pos] : undo.seqs) {
        const int32_t tail = cells[seq_id].tail;
        if (started || (tail >= 0 && pos <= cells[tail].pos)) {
            lose(seq_id);
        }
    }

    // between two ubatches the state of a cell lies in its own row (find_slot ends with src = i)
    for (uint32_t c = row0; c < std::min(row1, size); ++c) {
        for (const llama_seq_id seq_id : cells[c].seq_id) {
            lose(seq_id);
        }
    }

    for (const llama_seq_id seq_id : lost) {
        seq_rm(seq_id, -1, -1);
    }

    return lost;
}

void llama_memory_recurrent::ubatch_failed(bool started) {
    ubatch_restore(started);
}

void llama_memory_hybrid::ubatch_failed(bool started) {
    for (const llama_seq_id seq_id : mem_recr->ubatch_restore(started)) {
        seq_rm(seq_id, -1, -1);
    }
}

void llama_memory_hybrid_iswa::ubatch_failed(bool started) {
    for (const llama_seq_id seq_id : mem_recr->ubatch_restore(started)) {
        seq_rm(seq_id, -1, -1);
    }
}
