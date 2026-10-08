// the fork's KQ mask fill, included by llama-kv-cache.cpp after the upstream set_input_kq_mask_impl: the rows of
// the ubatch are filled in parallel (llama_parallel_for). LLAMA_UPSTREAM=kq-mask runs the upstream function

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl_fork(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // the rows are filled in parts, in parallel for a large mask. each part keeps its own bookkeeping, so it
        // fills its first row of a sequence in full and the next rows of that sequence copy it
        llama_parallel_for(n_tps, (int64_t) n_tps*n_kv, [&](int64_t ii0, int64_t ii1) {
            // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
            std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
            std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

            for (uint32_t ii = (uint32_t) ii0; ii < (uint32_t) ii1; ++ii) {
                const uint32_t i = s*n_tps + ii;

                const llama_seq_id seq_id = ubatch->seq_id[i][0];

                const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                      llama_pos p0 = -1;
                const llama_pos p1 = ubatch->pos[i];

                // for M-RoPE
                const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
                const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

                const uint64_t idst = n_kv*i;

                // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
                // the only cells that could change are the ones that are with similar positions as the
                //   ones in the batch (i.e. due to causal masking, SWA, etc.)
                // keep track of those cells and shortcut the loop to save time
                // note: this optimization is not compatible with Alibi position encoding
                // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
                bool prev = false;

                auto & idxs = seq_idxs[seq_id];

                if (!alibi) {
                    if (seq_srct.find(seq_id) != seq_srct.end()) {
                        const uint32_t srct = seq_srct[seq_id];

                        const uint64_t idst_prev = n_kv*srct;

                        std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                        prev = true;
                    } else {
                        idxs.clear();
                        idxs.reserve(ubatch->n_tokens + n_swa + 32);

                        seq_srct[seq_id] = i;
                    }
                }

                for (uint32_t jj = 0; jj < n_kv; ++jj) {
                    uint32_t j = jj;

                    // we have an exiting mask for this sequence -> update just seq_idxs
                    if (!alibi) {
                        if (prev) {
                            if (jj >= idxs.size()) {
                                break;
                            }

                            j = idxs[jj];
                        }
                    }

                    if (cells.is_empty(j)) {
                        goto skip;
                    }

                    // mask the token if not the same sequence
                    if (!cells.seq_has(j, seq_id)) {
                        goto skip;
                    }

                    p0 = cells.pos_get(j);

                    if (!alibi) {
                        if (!prev) {
                            // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                            if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                                idxs.push_back(j);
                            }
                        }
                    }

                    if (causal) {
                        // mask future tokens
                        if (p0 > p1) {
                            goto skip;
                        }

                        // M-RoPE causal mask
                        if (is_2d) {
                            if (p0 == p1) {
                                const auto & p0_ext = cells.ext_get(j);

                                if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                    goto skip;
                                }
                            }
                        }
                    }

                    // apply SWA if any
                    if (swa) {
                        // see llama_non_causal_type
                        const bool in_span = !causal && args.hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL && p0 >= seq_pos_min[seq_id];
                        if (!in_span && llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                            goto skip;
                        }
                    }

                    if (alibi) {
                        data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                    } else {
                        data[idst + j] = mask_keep;
                    }

                    continue;
    skip:
                    data[idst + j] = mask_drop;
                }
            }
        });
    }
}
