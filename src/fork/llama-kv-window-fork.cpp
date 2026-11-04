// the window cache of the dense MTP head of the fork's qwen4exp (llama_mtp_window_kv): room for an image that does not
// move the window (llama_kv_cache::swa_room_on_fork)

#include "llama-kv-cache.h"

#include "llama-batch.h"
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <vector>

bool llama_kv_cache::swa_room_fork(const llama_ubatch & ubatch, const std::vector<llama_ubatch> & ubatches) {
    // the cache keeps the window of every sequence plus one ubatch, a cell per position. the tokens of an image share
    // one temporal position (M-RoPE), so its cells come on top of the window's, and until the next position no cell of
    // the window is masked: a ubatch past the spare one finds no cell. the oldest cells of the sequences of the ubatch
    // go then - only cells below the lowest position of each sequence in the whole batch, so that no cell an earlier
    // ubatch of it holds is taken (prepare() places them one by one). the head attends to that many cells fewer at the
    // far end of its window until the window passes the image; the target, which checks every draft, does not change
    static std::atomic<bool> logged{false};

    bool freed = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        // the tokens of the ubatch in this stream and their sequences
        uint32_t                    n_tokens_s = 0;
        llama_kv_cells::seq_set_t   seqs;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (seq_to_stream[ubatch.seq_id[i][0]] != s) {
                continue;
            }
            n_tokens_s++;
            for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
                seqs.set(ubatch.seq_id[i][j]);
            }
        }
        if (n_tokens_s == 0) {
            continue;
        }

        auto & cells = v_cells[s];

        // enough empty cells: find_slot failed for another reason
        const uint32_t n_empty = cells.size() - cells.get_used();
        if (n_empty >= n_tokens_s) {
            continue;
        }

        // a few cells more than the ubatch lacks: after an image every step of the head would else search in vain
        // first, until the window passes the image (16384 positions by default)
        uint32_t n_need = n_tokens_s - n_empty + std::min<uint32_t>(256, n_swa/64);

        for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ && n_need > 0; ++seq_id) {
            if (!seqs.test(seq_id)) {
                continue;
            }

            // the lowest position of the sequence in the batch
            llama_pos p_lim = std::numeric_limits<llama_pos>::max();
            for (const auto & ub : ubatches) {
                for (uint32_t i = 0; i < ub.n_tokens; ++i) {
                    for (int32_t j = 0; j < ub.n_seq_id[i]; ++j) {
                        if (ub.seq_id[i][j] == seq_id) {
                            p_lim = std::min(p_lim, ub.pos[i]);
                        }
                    }
                }
            }

            // the oldest cells first, of this sequence alone (a cell a seq_cp left shared stays)
            std::vector<uint32_t> idxs;
            for (const auto & [pos, idx] : cells.seq_pos_get(seq_id)) {
                if (pos >= p_lim || idxs.size() >= n_need) {
                    break;
                }
                if (cells.seq_count(idx) == 1) {
                    idxs.push_back(idx);
                }
            }
            if (idxs.empty()) {
                continue;
            }

            const llama_pos p_first = cells.pos_get(idxs.front());
            const llama_pos p_last  = cells.pos_get(idxs.back());

            for (const uint32_t idx : idxs) {
                cells.seq_rm(idx, seq_id);
            }
            n_need -= (uint32_t) idxs.size();
            freed   = true;

            if (!logged.exchange(true)) {
                LLAMA_LOG_INFO("%s: window cache (window %u, %u cells): sequence %d holds more cells than positions (an "
                        "image), its %zu oldest cells (positions %d..%d) give way for a ubatch of %u; logged once\n",
                        __func__, n_swa, cells.size(), seq_id, idxs.size(), p_first, p_last, n_tokens_s);
            } else {
                LLAMA_LOG_DEBUG("%s: sequence %d gives up %zu cells (positions %d..%d) for a ubatch of %u\n", __func__,
                        seq_id, idxs.size(), p_first, p_last, n_tokens_s);
            }
        }
    }

    return freed;
}
