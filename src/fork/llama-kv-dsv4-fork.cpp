// a ubatch that fails on the DSV4 memory (llama_kv_cache_dsv4::ubatch_save, ubatch_failed)

#include "llama-kv-cache-dsv4.h"

#include "llama-batch.h"

#include <algorithm>
#include <vector>

void llama_kv_cache_dsv4::ubatch_save(const llama_ubatch & ubatch) {
    undo.valid  = true;
    undo.rs_idx = rs_idx;
    undo.rs_max = rs_max;

    // the first token of a sequence holds its lowest position: a batch whose positions go down is refused
    // (llama_batch_allocr::init). an entry without a position is not a token (as the other walks over a ubatch here)
    undo.seqs.clear();
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.pos[i] < 0) {
            continue;
        }
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];

            const bool seen = std::any_of(undo.seqs.begin(), undo.seqs.end(), [seq_id](const std::pair<llama_seq_id, llama_pos> & e) {
                return e.first == seq_id;
            });
            if (!seen) {
                undo.seqs.emplace_back(seq_id, ubatch.pos[i]);
            }
        }
    }

    update_rs_for_ubatches({ ubatch });
}

void llama_kv_cache_dsv4::ubatch_failed(bool started) {
    if (!undo.valid) {
        return;
    }
    undo.valid = false;

    rs_idx.swap(undo.rs_idx);
    rs_max.swap(undo.rs_max);

    for (const auto & [seq_id, pos] : undo.seqs) {
        if (started) {
            seq_rm(seq_id, -1, -1);
        } else {
            // the positions of a batch of tokens come after the ones the cache holds
            kv_raw->seq_rm(seq_id, pos, -1);
        }
    }
}
